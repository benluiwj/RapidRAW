#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod culling;
mod file_management;
mod formats;
mod gpu_processing;
mod image_loader;
mod image_processing;
mod lut_processing;
mod mask_generation;
mod panorama_stitching;
mod panorama_utils;
mod raw_processing;
mod tagging;

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Cursor, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use base64::{Engine as _, engine::general_purpose};
use chrono::{DateTime, Utc};
use image::codecs::jpeg::JpegEncoder;
use image::{
    DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Luma, RgbImage, Rgba, RgbaImage,
};
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::metadata::Metadata;
use little_exif::rational::uR64;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use tauri::{Emitter, Manager, ipc::Response};
use tempfile::NamedTempFile;
use tokio::task::JoinHandle;
use wgpu::{Texture, TextureView};

use crate::file_management::{AppSettings, get_sidecar_path, load_settings};
use crate::formats::is_raw_file;
use crate::image_loader::{
    composite_patches_on_image, load_and_composite, load_base_image_from_bytes,
};
use crate::image_processing::{
    Crop, GpuContext, ImageMetadata, apply_coarse_rotation, apply_crop, apply_flip, apply_rotation,
    get_all_adjustments_from_json, get_or_init_gpu_context, process_and_get_dynamic_image,
};
use crate::lut_processing::Lut;
use crate::mask_generation::{MaskDefinition, generate_mask_bitmap};

#[derive(Clone)]
pub struct LoadedImage {
    path: String,
    image: DynamicImage,
    full_width: u32,
    full_height: u32,
}

#[derive(Clone)]
pub struct CachedPreview {
    image: DynamicImage,
    transform_hash: u64,
    scale: f32,
    unscaled_crop_offset: (f32, f32),
}

pub struct GpuImageCache {
    pub texture: Texture,
    pub texture_view: TextureView,
    pub width: u32,
    pub height: u32,
    pub transform_hash: u64,
}

pub struct AppState {
    original_image: Mutex<Option<LoadedImage>>,
    cached_preview: Mutex<Option<CachedPreview>>,
    gpu_context: Mutex<Option<GpuContext>>,
    gpu_image_cache: Mutex<Option<GpuImageCache>>,
    export_task_handle: Mutex<Option<JoinHandle<()>>>,
    panorama_result: Arc<Mutex<Option<RgbImage>>>,
    indexing_task_handle: Mutex<Option<JoinHandle<()>>>,
    pub lut_cache: Mutex<HashMap<String, Arc<Lut>>>,
    thumbnail_cancellation_token: Arc<AtomicBool>,
}

#[derive(serde::Serialize)]
struct LoadImageResult {
    #[serde(with = "serde_bytes")]
    original_image_bytes: Vec<u8>,
    width: u32,
    height: u32,
    metadata: ImageMetadata,
    exif: HashMap<String, String>,
    is_raw: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
enum ResizeMode {
    LongEdge,
    Width,
    Height,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct ResizeOptions {
    mode: ResizeMode,
    value: u32,
    dont_enlarge: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct ExportSettings {
    jpeg_quality: u8,
    resize: Option<ResizeOptions>,
    keep_metadata: bool,
    strip_gps: bool,
    filename_template: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GitHubContent {
    name: String,
    path: String,
    download_url: String,
    #[serde(rename = "type")]
    content_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommunityPreset {
    pub name: String,
    pub creator: String,
    pub adjustments: Value,
}

#[derive(Serialize)]
struct LutParseResult {
    size: u32,
}

fn apply_all_transformations(
    image: &DynamicImage,
    adjustments: &serde_json::Value,
    scale: f32,
) -> (DynamicImage, (f32, f32)) {
    let orientation_steps = adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
    let rotation_degrees = adjustments["rotation"].as_f64().unwrap_or(0.0) as f32;
    let flip_horizontal = adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    let flip_vertical = adjustments["flipVertical"].as_bool().unwrap_or(false);

    let coarse_rotated_image = apply_coarse_rotation(image.clone(), orientation_steps);
    let flipped_image = apply_flip(coarse_rotated_image, flip_horizontal, flip_vertical);
    let rotated_image = apply_rotation(&flipped_image, rotation_degrees);

    let crop_data: Option<Crop> = serde_json::from_value(adjustments["crop"].clone()).ok();

    let scaled_crop_json = if let Some(c) = &crop_data {
        serde_json::to_value(Crop {
            x: c.x * scale as f64,
            y: c.y * scale as f64,
            width: c.width * scale as f64,
            height: c.height * scale as f64,
        })
        .unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };

    let cropped_image = apply_crop(rotated_image, &scaled_crop_json);

    let unscaled_crop_offset = crop_data.map_or((0.0, 0.0), |c| (c.x as f32, c.y as f32));

    (cropped_image, unscaled_crop_offset)
}

fn calculate_transform_hash(adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();

    let orientation_steps = adjustments["orientationSteps"].as_u64().unwrap_or(0);
    orientation_steps.hash(&mut hasher);

    let rotation = adjustments["rotation"].as_f64().unwrap_or(0.0);
    (rotation.to_bits()).hash(&mut hasher);

    let flip_h = adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    flip_h.hash(&mut hasher);

    let flip_v = adjustments["flipVertical"].as_bool().unwrap_or(false);
    flip_v.hash(&mut hasher);

    if let Some(crop_val) = adjustments.get("crop") {
        if !crop_val.is_null() {
            crop_val.to_string().hash(&mut hasher);
        }
    }

    if let Some(patches_val) = adjustments.get("aiPatches") {
        if let Some(patches_arr) = patches_val.as_array() {
            patches_arr.len().hash(&mut hasher);

            for patch in patches_arr {
                if let Some(id) = patch.get("id").and_then(|v| v.as_str()) {
                    id.hash(&mut hasher);
                }

                let is_visible = patch
                    .get("visible")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                is_visible.hash(&mut hasher);

                if let Some(patch_data) = patch.get("patchData") {
                    let color_len = patch_data
                        .get("color")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .len();
                    color_len.hash(&mut hasher);

                    let mask_len = patch_data
                        .get("mask")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .len();
                    mask_len.hash(&mut hasher);
                } else {
                    let data_len = patch
                        .get("patchDataBase64")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .len();
                    data_len.hash(&mut hasher);
                }

                if let Some(sub_masks_val) = patch.get("subMasks") {
                    sub_masks_val.to_string().hash(&mut hasher);
                }

                let invert = patch
                    .get("invert")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                invert.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

fn calculate_full_job_hash(path: &str, adjustments: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    adjustments.to_string().hash(&mut hasher);
    hasher.finish()
}

fn generate_transformed_preview(
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
    app_handle: &tauri::AppHandle,
) -> Result<(DynamicImage, f32, (f32, f32)), String> {
    let patched_original_image = composite_patches_on_image(&loaded_image.image, adjustments)
        .map_err(|e| format!("Failed to composite AI patches: {}", e))?;

    let (full_w, full_h) = (loaded_image.full_width, loaded_image.full_height);

    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let final_preview_dim = settings.editor_preview_resolution.unwrap_or(1920);

    let (processing_base, scale_for_gpu) =
        if full_w > final_preview_dim || full_h > final_preview_dim {
            let base = patched_original_image.thumbnail(final_preview_dim, final_preview_dim);
            let scale = if full_w > 0 {
                base.width() as f32 / full_w as f32
            } else {
                1.0
            };
            (base, scale)
        } else {
            (patched_original_image.clone(), 1.0)
        };

    let (final_preview_base, unscaled_crop_offset) =
        apply_all_transformations(&processing_base, adjustments, scale_for_gpu);

    Ok((final_preview_base, scale_for_gpu, unscaled_crop_offset))
}

fn read_exif_data(file_bytes: &[u8]) -> HashMap<String, String> {
    let mut exif_data = HashMap::new();
    let exif_reader = exif::Reader::new();
    if let Ok(exif) = exif_reader.read_from_container(&mut Cursor::new(file_bytes)) {
        for field in exif.fields() {
            exif_data.insert(
                field.tag.to_string(),
                field.display_value().with_unit(&exif).to_string(),
            );
        }
    }
    exif_data
}

fn get_or_load_lut(state: &tauri::State<AppState>, path: &str) -> Result<Arc<Lut>, String> {
    let mut cache = state.lut_cache.lock().unwrap();
    if let Some(lut) = cache.get(path) {
        return Ok(lut.clone());
    }

    let lut = lut_processing::parse_lut_file(path).map_err(|e| e.to_string())?;
    let arc_lut = Arc::new(lut);
    cache.insert(path.to_string(), arc_lut.clone());
    Ok(arc_lut)
}

#[tauri::command]
async fn load_image(
    path: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<LoadImageResult, String> {
    let sidecar_path = get_sidecar_path(&path);
    let metadata: ImageMetadata = if sidecar_path.exists() {
        let file_content = fs::read_to_string(sidecar_path).map_err(|e| e.to_string())?;
        serde_json::from_str(&file_content).unwrap_or_default()
    } else {
        ImageMetadata::default()
    };

    let file_bytes = fs::read(&path).map_err(|e| e.to_string())?;
    let pristine_img =
        load_base_image_from_bytes(&file_bytes, &path, false).map_err(|e| e.to_string())?;

    let (orig_width, orig_height) = pristine_img.dimensions();
    let is_raw = is_raw_file(&path);

    let exif_data = read_exif_data(&file_bytes);

    let settings = load_settings(app_handle).unwrap_or_default();
    let display_preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
    let display_preview = pristine_img.thumbnail(display_preview_dim, display_preview_dim);

    let mut buf = Cursor::new(Vec::new());
    display_preview
        .to_rgb8()
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
        .map_err(|e| e.to_string())?;
    let original_image_bytes = buf.into_inner();

    *state.cached_preview.lock().unwrap() = None;
    *state.gpu_image_cache.lock().unwrap() = None;
    *state.original_image.lock().unwrap() = Some(LoadedImage {
        path: path.clone(),
        image: pristine_img,
        full_width: orig_width,
        full_height: orig_height,
    });

    Ok(LoadImageResult {
        original_image_bytes,
        width: orig_width,
        height: orig_height,
        metadata,
        exif: exif_data,
        is_raw,
    })
}

#[tauri::command]
fn cancel_thumbnail_generation(state: tauri::State<AppState>) -> Result<(), String> {
    state
        .thumbnail_cancellation_token
        .store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn apply_adjustments(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let context = get_or_init_gpu_context(&state)?;
    let adjustments_clone = js_adjustments.clone();

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;
    let new_transform_hash = calculate_transform_hash(&adjustments_clone);

    let mut cached_preview_lock = state.cached_preview.lock().unwrap();

    let (final_preview_base, scale_for_gpu, unscaled_crop_offset) =
        if let Some(cached) = &*cached_preview_lock {
            if cached.transform_hash == new_transform_hash {
                (
                    cached.image.clone(),
                    cached.scale,
                    cached.unscaled_crop_offset,
                )
            } else {
                *state.gpu_image_cache.lock().unwrap() = None;
                let (base, scale, offset) =
                    generate_transformed_preview(&loaded_image, &adjustments_clone, &app_handle)?;
                *cached_preview_lock = Some(CachedPreview {
                    image: base.clone(),
                    transform_hash: new_transform_hash,
                    scale,
                    unscaled_crop_offset: offset,
                });
                (base, scale, offset)
            }
        } else {
            *state.gpu_image_cache.lock().unwrap() = None;
            let (base, scale, offset) =
                generate_transformed_preview(&loaded_image, &adjustments_clone, &app_handle)?;
            *cached_preview_lock = Some(CachedPreview {
                image: base.clone(),
                transform_hash: new_transform_hash,
                scale,
                unscaled_crop_offset: offset,
            });
            (base, scale, offset)
        };

    drop(cached_preview_lock);

    thread::spawn(move || {
        let state = app_handle.state::<AppState>();
        let (preview_width, preview_height) = final_preview_base.dimensions();

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_else(Vec::new);

        let scaled_crop_offset = (
            unscaled_crop_offset.0 * scale_for_gpu,
            unscaled_crop_offset.1 * scale_for_gpu,
        );

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                generate_mask_bitmap(
                    def,
                    preview_width,
                    preview_height,
                    scale_for_gpu,
                    scaled_crop_offset,
                )
            })
            .collect();

        let final_adjustments = get_all_adjustments_from_json(&adjustments_clone);
        let lut_path = adjustments_clone["lutPath"].as_str();
        let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

        if let Ok(final_processed_image) = process_and_get_dynamic_image(
            &context,
            &state,
            &final_preview_base,
            new_transform_hash,
            final_adjustments,
            &mask_bitmaps,
            lut,
        ) {
            if let Ok(histogram_data) =
                image_processing::calculate_histogram_from_image(&final_processed_image)
            {
                let _ = app_handle.emit("histogram-update", histogram_data);
            }

            if let Ok(waveform_data) =
                image_processing::calculate_waveform_from_image(&final_processed_image)
            {
                let _ = app_handle.emit("waveform-update", waveform_data);
            }

            let mut buf = Cursor::new(Vec::new());
            if final_processed_image
                .to_rgb8()
                .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
                .is_ok()
            {
                let _ = app_handle.emit("preview-update-final", buf.get_ref());
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn generate_uncropped_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let context = get_or_init_gpu_context(&state)?;
    let adjustments_clone = js_adjustments.clone();
    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    thread::spawn(move || {
        let state = app_handle.state::<AppState>();
        let path = loaded_image.path.clone();
        let unique_hash = calculate_full_job_hash(&path, &adjustments_clone);
        let patched_image =
            match composite_patches_on_image(&loaded_image.image, &adjustments_clone) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("Failed to composite patches for uncropped preview: {}", e);
                    loaded_image.image
                }
            };

        let orientation_steps = adjustments_clone["orientationSteps"].as_u64().unwrap_or(0) as u8;
        let coarse_rotated_image = apply_coarse_rotation(patched_image, orientation_steps);

        let settings = load_settings(app_handle.clone()).unwrap_or_default();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);

        let (rotated_w, rotated_h) = coarse_rotated_image.dimensions();

        let (processing_base, scale_for_gpu) = if rotated_w > preview_dim || rotated_h > preview_dim
        {
            let base = coarse_rotated_image.thumbnail(preview_dim, preview_dim);
            let scale = if rotated_w > 0 {
                base.width() as f32 / rotated_w as f32
            } else {
                1.0
            };
            (base, scale)
        } else {
            (coarse_rotated_image.clone(), 1.0)
        };

        let (preview_width, preview_height) = processing_base.dimensions();

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_else(Vec::new);

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                generate_mask_bitmap(
                    def,
                    preview_width,
                    preview_height,
                    scale_for_gpu,
                    (0.0, 0.0),
                )
            })
            .collect();

        let uncropped_adjustments = get_all_adjustments_from_json(&adjustments_clone);
        let lut_path = adjustments_clone["lutPath"].as_str();
        let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

        if let Ok(processed_image) = process_and_get_dynamic_image(
            &context,
            &state,
            &processing_base,
            unique_hash,
            uncropped_adjustments,
            &mask_bitmaps,
            lut,
        ) {
            let mut buf = Cursor::new(Vec::new());
            if processed_image
                .to_rgb8()
                .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
                .is_ok()
            {
                let _ = app_handle.emit("preview-update-uncropped", buf.get_ref());
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn generate_original_transformed_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded")?;

    let settings = load_settings(app_handle).unwrap_or_default();
    let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
    let preview_base = loaded_image.image.thumbnail(preview_dim, preview_dim);
    let scale = if loaded_image.full_width > 0 {
        preview_base.width() as f32 / loaded_image.full_width as f32
    } else {
        1.0
    };

    let (transformed_image, _unscaled_crop_offset) =
        apply_all_transformations(&preview_base, &js_adjustments, scale);

    let mut buf = Cursor::new(Vec::new());
    transformed_image
        .to_rgb8()
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
        .map_err(|e| e.to_string())?;

    Ok(Response::new(buf.into_inner()))
}

fn get_full_image_for_processing(state: &tauri::State<AppState>) -> Result<DynamicImage, String> {
    let original_image_lock = state.original_image.lock().unwrap();
    let loaded_image = original_image_lock
        .as_ref()
        .ok_or("No original image loaded")?;
    Ok(loaded_image.image.clone())
}

#[tauri::command]
fn generate_fullscreen_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
) -> Result<Response, String> {
    let context = get_or_init_gpu_context(&state)?;
    let original_image = get_full_image_for_processing(&state)?;
    let path = state
        .original_image
        .lock()
        .unwrap()
        .as_ref()
        .ok_or("Original image path not found")?
        .path
        .clone();
    let unique_hash = calculate_full_job_hash(&path, &js_adjustments);
    let base_image = composite_patches_on_image(&original_image, &js_adjustments)
        .map_err(|e| format!("Failed to composite AI patches for fullscreen: {}", e))?;

    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&base_image, &js_adjustments, 1.0);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_else(Vec::new);

    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset))
        .collect();

    let all_adjustments = get_all_adjustments_from_json(&js_adjustments);
    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

    let final_image = process_and_get_dynamic_image(
        &context,
        &state,
        &transformed_image,
        unique_hash,
        all_adjustments,
        &mask_bitmaps,
        lut,
    )?;

    let mut buf = Cursor::new(Vec::new());
    final_image
        .to_rgb8()
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 92))
        .map_err(|e| e.to_string())?;

    Ok(Response::new(buf.into_inner()))
}

fn process_image_for_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
) -> Result<DynamicImage, String> {
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&base_image, &js_adjustments, 1.0);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_else(Vec::new);

    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset))
        .collect();

    let mut all_adjustments = get_all_adjustments_from_json(&js_adjustments);
    all_adjustments.global.show_clipping = 0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

    let unique_hash = calculate_full_job_hash(path, js_adjustments);

    let mut final_image = process_and_get_dynamic_image(
        &context,
        &state,
        &transformed_image,
        unique_hash,
        all_adjustments,
        &mask_bitmaps,
        lut,
    )?;

    if let Some(resize_opts) = &export_settings.resize {
        let (current_w, current_h) = final_image.dimensions();
        let should_resize = if resize_opts.dont_enlarge {
            match resize_opts.mode {
                ResizeMode::LongEdge => current_w.max(current_h) > resize_opts.value,
                ResizeMode::Width => current_w > resize_opts.value,
                ResizeMode::Height => current_h > resize_opts.value,
            }
        } else {
            true
        };

        if should_resize {
            final_image = match resize_opts.mode {
                ResizeMode::LongEdge => {
                    let (w, h) = if current_w > current_h {
                        (
                            resize_opts.value,
                            (resize_opts.value as f32 * (current_h as f32 / current_w as f32))
                                .round() as u32,
                        )
                    } else {
                        (
                            (resize_opts.value as f32 * (current_w as f32 / current_h as f32))
                                .round() as u32,
                            resize_opts.value,
                        )
                    };
                    final_image.thumbnail(w, h)
                }
                ResizeMode::Width => final_image.thumbnail(resize_opts.value, u32::MAX),
                ResizeMode::Height => final_image.thumbnail(u32::MAX, resize_opts.value),
            };
        }
    }
    Ok(final_image)
}

fn encode_image_to_bytes(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
) -> Result<Vec<u8>, String> {
    let mut image_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut image_bytes);

    match output_format.to_lowercase().as_str() {
        "jpg" | "jpeg" => {
            let rgb_image = image.to_rgb8();
            let encoder = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
            rgb_image
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "png" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
        }
        "tiff" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Tiff)
                .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("Unsupported file format: {}", output_format)),
    };
    Ok(image_bytes)
}

#[tauri::command]
async fn export_image(
    original_path: String,
    output_path: String,
    js_adjustments: Value,
    export_settings: ExportSettings,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    if state.export_task_handle.lock().unwrap().is_some() {
        return Err("An export is already in progress.".to_string());
    }

    let context = get_or_init_gpu_context(&state)?;
    let original_image_data = get_full_image_for_processing(&state)?;
    let context = Arc::new(context);

    let task = tokio::spawn(async move {
        let state = app_handle.state::<AppState>();
        let processing_result: Result<(), String> = (|| {
            let base_image = composite_patches_on_image(&original_image_data, &js_adjustments)
                .map_err(|e| format!("Failed to composite AI patches for export: {}", e))?;

            let final_image = process_image_for_export(
                &original_path,
                &base_image,
                &js_adjustments,
                &export_settings,
                &context,
                &state,
            )?;

            let output_path_obj = std::path::Path::new(&output_path);
            let extension = output_path_obj
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();

            let mut image_bytes =
                encode_image_to_bytes(&final_image, &extension, export_settings.jpeg_quality)?;

            write_image_with_metadata(
                &mut image_bytes,
                &original_path,
                &extension,
                export_settings.keep_metadata,
                export_settings.strip_gps,
            )?;

            fs::write(&output_path, image_bytes).map_err(|e| e.to_string())?;

            Ok(())
        })();

        if let Err(e) = processing_result {
            let _ = app_handle.emit("export-error", e);
        } else {
            let _ = app_handle.emit("export-complete", ());
        }

        *app_handle
            .state::<AppState>()
            .export_task_handle
            .lock()
            .unwrap() = None;
    });

    *state.export_task_handle.lock().unwrap() = Some(task);
    Ok(())
}

#[tauri::command]
async fn batch_export_images(
    output_folder: String,
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    if state.export_task_handle.lock().unwrap().is_some() {
        return Err("An export is already in progress.".to_string());
    }

    let context = get_or_init_gpu_context(&state)?;
    let context = Arc::new(context);

    let task = tokio::spawn(async move {
        let state = app_handle.state::<AppState>();
        let output_folder_path = std::path::Path::new(&output_folder);
        let total_paths = paths.len();

        for (i, image_path_str) in paths.iter().enumerate() {
            if app_handle
                .state::<AppState>()
                .export_task_handle
                .lock()
                .unwrap()
                .is_none()
            {
                println!("Export cancelled during batch processing.");
                let _ = app_handle.emit("export-cancelled", ());
                return;
            }

            let _ = app_handle.emit(
                "batch-export-progress",
                serde_json::json!({ "current": i, "total": total_paths, "path": image_path_str }),
            );

            let processing_result: Result<(), String> = (|| {
                let sidecar_path = get_sidecar_path(image_path_str);
                let metadata: ImageMetadata = if sidecar_path.exists() {
                    let file_content =
                        fs::read_to_string(sidecar_path).map_err(|e| e.to_string())?;
                    serde_json::from_str(&file_content).unwrap_or_default()
                } else {
                    ImageMetadata::default()
                };
                let js_adjustments = metadata.adjustments;

                let base_image = load_and_composite(image_path_str, &js_adjustments, false)
                    .map_err(|e| e.to_string())?;

                let final_image = process_image_for_export(
                    image_path_str,
                    &base_image,
                    &js_adjustments,
                    &export_settings,
                    &context,
                    &state,
                )?;

                let original_path = std::path::Path::new(image_path_str);

                let file_date: DateTime<Utc> = Metadata::new_from_path(original_path)
                    .ok()
                    .and_then(|metadata| {
                        metadata
                            .get_tag(&ExifTag::DateTimeOriginal("".to_string()))
                            .next()
                            .and_then(|tag| {
                                if let &ExifTag::DateTimeOriginal(ref dt_str) = tag {
                                    chrono::NaiveDateTime::parse_from_str(
                                        dt_str,
                                        "%Y:%m:%d %H:%M:%S",
                                    )
                                    .ok()
                                    .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                                } else {
                                    None
                                }
                            })
                    })
                    .unwrap_or_else(|| {
                        fs::metadata(original_path)
                            .ok()
                            .and_then(|m| m.created().ok())
                            .map(DateTime::<Utc>::from)
                            .unwrap_or_else(Utc::now)
                    });

                let filename_template = export_settings
                    .filename_template
                    .as_deref()
                    .unwrap_or("{original_filename}_edited");
                let new_stem = crate::file_management::generate_filename_from_template(
                    filename_template,
                    original_path,
                    i + 1,
                    total_paths,
                    &file_date,
                );
                let new_filename = format!("{}.{}", new_stem, output_format);
                let output_path = output_folder_path.join(new_filename);

                let mut image_bytes = encode_image_to_bytes(
                    &final_image,
                    &output_format,
                    export_settings.jpeg_quality,
                )?;

                write_image_with_metadata(
                    &mut image_bytes,
                    image_path_str,
                    &output_format,
                    export_settings.keep_metadata,
                    export_settings.strip_gps,
                )?;

                fs::write(&output_path, image_bytes).map_err(|e| e.to_string())?;

                Ok(())
            })();

            if let Err(e) = processing_result {
                eprintln!("Failed to export {}: {}", image_path_str, e);
                let _ = app_handle.emit("export-error", e);
                *app_handle
                    .state::<AppState>()
                    .export_task_handle
                    .lock()
                    .unwrap() = None;
                return;
            }
        }

        let _ = app_handle.emit(
            "batch-export-progress",
            serde_json::json!({ "current": total_paths, "total": total_paths, "path": "" }),
        );
        let _ = app_handle.emit("export-complete", ());
        *app_handle
            .state::<AppState>()
            .export_task_handle
            .lock()
            .unwrap() = None;
    });

    *state.export_task_handle.lock().unwrap() = Some(task);
    Ok(())
}

#[tauri::command]
fn cancel_export(state: tauri::State<AppState>) -> Result<(), String> {
    match state.export_task_handle.lock().unwrap().take() {
        Some(handle) => {
            handle.abort();
            println!("Export task cancellation requested.");
        }
        _ => {
            return Err("No export task is currently running.".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
async fn estimate_export_size(
    js_adjustments: Value,
    export_settings: ExportSettings,
    output_format: String,
    state: tauri::State<'_, AppState>,
) -> Result<usize, String> {
    let context = get_or_init_gpu_context(&state)?;
    let original_image_data = get_full_image_for_processing(&state)?;
    let path = state
        .original_image
        .lock()
        .unwrap()
        .as_ref()
        .ok_or("Original image path not found")?
        .path
        .clone();

    let base_image = composite_patches_on_image(&original_image_data, &js_adjustments)
        .map_err(|e| format!("Failed to composite AI patches for estimation: {}", e))?;

    let final_image = process_image_for_export(
        &path,
        &base_image,
        &js_adjustments,
        &export_settings,
        &context,
        &state,
    )?;

    let image_bytes =
        encode_image_to_bytes(&final_image, &output_format, export_settings.jpeg_quality)?;

    Ok(image_bytes.len())
}

#[tauri::command]
async fn estimate_batch_export_size(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    state: tauri::State<'_, AppState>,
) -> Result<usize, String> {
    if paths.is_empty() {
        return Ok(0);
    }
    let context = get_or_init_gpu_context(&state)?;
    let first_path = &paths[0];

    let sidecar_path = get_sidecar_path(first_path);
    let metadata: ImageMetadata = if sidecar_path.exists() {
        let file_content = fs::read_to_string(sidecar_path).map_err(|e| e.to_string())?;
        serde_json::from_str(&file_content).unwrap_or_default()
    } else {
        ImageMetadata::default()
    };
    let js_adjustments = metadata.adjustments;

    let base_image =
        load_and_composite(first_path, &js_adjustments, false).map_err(|e| e.to_string())?;

    let final_image = process_image_for_export(
        first_path,
        &base_image,
        &js_adjustments,
        &export_settings,
        &context,
        &state,
    )?;

    let image_bytes =
        encode_image_to_bytes(&final_image, &output_format, export_settings.jpeg_quality)?;

    Ok(image_bytes.len() * paths.len())
}

fn write_image_with_metadata(
    image_bytes: &mut Vec<u8>,
    original_path_str: &str,
    output_format: &str,
    keep_metadata: bool,
    strip_gps: bool,
) -> Result<(), String> {
    if !keep_metadata || output_format.to_lowercase() == "tiff" {
        // FIXME: temporary solution until I find a way to write metadata to TIFF
        return Ok(());
    }

    let file_type = match output_format.to_lowercase().as_str() {
        "jpg" | "jpeg" => FileExtension::JPEG,
        "png" => FileExtension::PNG {
            as_zTXt_chunk: true,
        },
        "tiff" => FileExtension::TIFF,
        _ => return Ok(()),
    };

    let original_path = std::path::Path::new(original_path_str);
    if !original_path.exists() {
        eprintln!(
            "Original file not found, cannot copy metadata: {}",
            original_path_str
        );
        return Ok(());
    }

    if let Ok(mut metadata) = Metadata::new_from_path(original_path) {
        if strip_gps {
            let dummy_rational = uR64 {
                nominator: 0,
                denominator: 1,
            };
            let dummy_rational_vec1 = vec![dummy_rational.clone()];
            let dummy_rational_vec3 = vec![
                dummy_rational.clone(),
                dummy_rational.clone(),
                dummy_rational.clone(),
            ];

            metadata.remove_tag(ExifTag::GPSVersionID([0, 0, 0, 0].to_vec()));
            metadata.remove_tag(ExifTag::GPSLatitudeRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSLatitude(dummy_rational_vec3.clone()));
            metadata.remove_tag(ExifTag::GPSLongitudeRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSLongitude(dummy_rational_vec3.clone()));
            metadata.remove_tag(ExifTag::GPSAltitudeRef(vec![0]));
            metadata.remove_tag(ExifTag::GPSAltitude(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSTimeStamp(dummy_rational_vec3.clone()));
            metadata.remove_tag(ExifTag::GPSSatellites("".to_string()));
            metadata.remove_tag(ExifTag::GPSStatus("".to_string()));
            metadata.remove_tag(ExifTag::GPSMeasureMode("".to_string()));
            metadata.remove_tag(ExifTag::GPSDOP(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSSpeedRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSSpeed(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSTrackRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSTrack(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSImgDirectionRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSImgDirection(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSMapDatum("".to_string()));
            metadata.remove_tag(ExifTag::GPSDestLatitudeRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSDestLatitude(dummy_rational_vec3.clone()));
            metadata.remove_tag(ExifTag::GPSDestLongitudeRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSDestLongitude(dummy_rational_vec3.clone()));
            metadata.remove_tag(ExifTag::GPSDestBearingRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSDestBearing(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSDestDistanceRef("".to_string()));
            metadata.remove_tag(ExifTag::GPSDestDistance(dummy_rational_vec1.clone()));
            metadata.remove_tag(ExifTag::GPSProcessingMethod(vec![]));
            metadata.remove_tag(ExifTag::GPSAreaInformation(vec![]));
            metadata.remove_tag(ExifTag::GPSDateStamp("".to_string()));
            metadata.remove_tag(ExifTag::GPSDifferential(vec![0u16]));
            metadata.remove_tag(ExifTag::GPSHPositioningError(dummy_rational_vec1.clone()));
        }

        metadata.set_tag(ExifTag::Orientation(vec![1u16]));

        if metadata.write_to_vec(image_bytes, file_type).is_err() {
            eprintln!(
                "Failed to write metadata to image vector for {}",
                original_path_str
            );
        }
    } else {
        eprintln!(
            "Failed to read metadata from original file: {}",
            original_path_str
        );
    }

    Ok(())
}

#[tauri::command]
fn generate_mask_overlay(
    mask_def: MaskDefinition,
    width: u32,
    height: u32,
    scale: f32,
    crop_offset: (f32, f32),
) -> Result<String, String> {
    let scaled_crop_offset = (crop_offset.0 * scale, crop_offset.1 * scale);

    if let Some(gray_mask) =
        generate_mask_bitmap(&mask_def, width, height, scale, scaled_crop_offset)
    {
        let mut rgba_mask = RgbaImage::new(width, height);
        for (x, y, pixel) in gray_mask.enumerate_pixels() {
            let intensity = pixel[0];
            let alpha = (intensity as f32 * 0.5) as u8;
            rgba_mask.put_pixel(x, y, Rgba([255, 0, 0, alpha]));
        }

        let mut buf = Cursor::new(Vec::new());
        rgba_mask
            .write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| e.to_string())?;

        let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
        let data_url = format!("data:image/png;base64,{}", base64_str);

        Ok(data_url)
    } else {
        Ok("".to_string())
    }
}

#[tauri::command]
fn generate_preset_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
) -> Result<Response, String> {
    let context = get_or_init_gpu_context(&state)?;

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded for preset preview")?;
    let original_image = loaded_image.image;
    let path = loaded_image.path;
    let unique_hash = calculate_full_job_hash(&path, &js_adjustments);

    const PRESET_PREVIEW_DIM: u32 = 200;
    let preview_base = original_image.thumbnail(PRESET_PREVIEW_DIM, PRESET_PREVIEW_DIM);

    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&preview_base, &js_adjustments, 1.0);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_else(Vec::new);

    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset))
        .collect();

    let all_adjustments = get_all_adjustments_from_json(&js_adjustments);
    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

    let processed_image = process_and_get_dynamic_image(
        &context,
        &state,
        &transformed_image,
        unique_hash,
        all_adjustments,
        &mask_bitmaps,
        lut,
    )?;

    let mut buf = Cursor::new(Vec::new());
    processed_image
        .to_rgb8()
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 50))
        .map_err(|e| e.to_string())?;

    Ok(Response::new(buf.into_inner()))
}

#[tauri::command]
fn update_window_effect(theme: String, window: tauri::Window) {
    apply_window_effect(theme, window);
}

#[tauri::command]
fn get_supported_file_types() -> Result<serde_json::Value, String> {
    let raw_extensions: Vec<&str> = crate::formats::RAW_EXTENSIONS
        .iter()
        .map(|(ext, _)| *ext)
        .collect();
    let non_raw_extensions: Vec<&str> = crate::formats::NON_RAW_EXTENSIONS.to_vec();

    Ok(serde_json::json!({
        "raw": raw_extensions,
        "nonRaw": non_raw_extensions
    }))
}

#[tauri::command]
async fn stitch_panorama(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to stitch.".to_string());
    }

    let panorama_result_handle = state.panorama_result.clone();

    let task = tokio::task::spawn_blocking(move || {
        let panorama_result = panorama_stitching::stitch_images(paths, app_handle.clone());

        match panorama_result {
            Ok(panorama_image) => {
                let _ = app_handle.emit("panorama-progress", "Creating preview...");

                let (w, h) = panorama_image.dimensions();
                let (new_w, new_h) = if w > h {
                    (800, (800.0 * h as f32 / w as f32).round() as u32)
                } else {
                    ((800.0 * w as f32 / h as f32).round() as u32, 800)
                };
                let preview_image = image::imageops::resize(
                    &panorama_image,
                    new_w,
                    new_h,
                    image::imageops::FilterType::Triangle,
                );

                let mut buf = Cursor::new(Vec::new());

                if let Err(e) = preview_image.write_to(&mut buf, ImageFormat::Png) {
                    return Err(format!("Failed to encode panorama preview: {}", e));
                }

                let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
                let final_base64 = format!("data:image/png;base64,{}", base64_str);

                *panorama_result_handle.lock().unwrap() = Some(panorama_image);

                let _ = app_handle.emit(
                    "panorama-complete",
                    serde_json::json!({
                        "base64": final_base64,
                    }),
                );
                Ok(())
            }
            Err(e) => {
                let _ = app_handle.emit("panorama-error", e.clone());
                Err(e)
            }
        }
    });

    match task.await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(join_err) => Err(format!("Panorama task failed: {}", join_err)),
    }
}

#[tauri::command]
async fn fetch_community_presets() -> Result<Vec<CommunityPreset>, String> {
    let client = reqwest::Client::new();
    let url = "https://raw.githubusercontent.com/CyberTimon/RapidRAW-Presets/main/manifest.json";

    let response = client
        .get(url)
        .header("User-Agent", "RapidRAW-App")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch manifest from GitHub: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("GitHub returned an error: {}", response.status()));
    }

    let presets: Vec<CommunityPreset> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse manifest.json: {}", e))?;

    Ok(presets)
}

#[tauri::command]
async fn generate_all_community_previews(
    image_paths: Vec<String>,
    presets: Vec<CommunityPreset>,
    state: tauri::State<'_, AppState>,
) -> Result<HashMap<String, Vec<u8>>, String> {
    let context = crate::image_processing::get_or_init_gpu_context(&state)?;
    let mut results: HashMap<String, Vec<u8>> = HashMap::new();

    const TILE_DIM: u32 = 360;
    const PROCESSING_DIM: u32 = TILE_DIM * 2;

    let mut base_thumbnails: Vec<DynamicImage> = Vec::new();
    for image_path in image_paths.iter() {
        let image_bytes = fs::read(image_path).map_err(|e| e.to_string())?;
        let original_image =
            crate::image_loader::load_base_image_from_bytes(&image_bytes, &image_path, true)
                .map_err(|e| e.to_string())?;
        base_thumbnails.push(original_image.thumbnail(PROCESSING_DIM, PROCESSING_DIM));
    }

    for preset in presets.iter() {
        let mut processed_tiles: Vec<RgbImage> = Vec::new();
        let js_adjustments = &preset.adjustments;

        let mut preset_hasher = DefaultHasher::new();
        preset.name.hash(&mut preset_hasher);
        let preset_hash = preset_hasher.finish();

        for (i, base_image) in base_thumbnails.iter().enumerate() {
            let (transformed_image, unscaled_crop_offset) =
                crate::apply_all_transformations(&base_image, &js_adjustments, 1.0);
            let (img_w, img_h) = transformed_image.dimensions();

            let mask_definitions: Vec<MaskDefinition> = js_adjustments
                .get("masks")
                .and_then(|m| serde_json::from_value(m.clone()).ok())
                .unwrap_or_else(Vec::new);

            let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
                .iter()
                .filter_map(|def| {
                    generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset)
                })
                .collect();

            let all_adjustments = get_all_adjustments_from_json(&js_adjustments);
            let lut_path = js_adjustments["lutPath"].as_str();
            let lut = lut_path.and_then(|p| get_or_load_lut(&state, p).ok());

            let unique_hash = preset_hash.wrapping_add(i as u64);

            let processed_image_dynamic = crate::image_processing::process_and_get_dynamic_image(
                &context,
                &state,
                &transformed_image,
                unique_hash,
                all_adjustments,
                &mask_bitmaps,
                lut,
            )?;

            let processed_image = processed_image_dynamic.to_rgb8();

            let (proc_w, proc_h) = processed_image.dimensions();
            let size = proc_w.min(proc_h);
            let cropped_processed_image = image::imageops::crop_imm(
                &processed_image,
                (proc_w - size) / 2,
                (proc_h - size) / 2,
                size,
                size,
            )
            .to_image();

            let final_tile = image::imageops::resize(
                &cropped_processed_image,
                TILE_DIM,
                TILE_DIM,
                image::imageops::FilterType::Lanczos3,
            );
            processed_tiles.push(final_tile);
        }

        let final_image_buffer = match processed_tiles.len() {
            1 => processed_tiles.remove(0),
            2 => {
                let mut canvas = RgbImage::new(TILE_DIM * 2, TILE_DIM);
                image::imageops::overlay(&mut canvas, &processed_tiles[0], 0, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[1], TILE_DIM as i64, 0);
                canvas
            }
            4 => {
                let mut canvas = RgbImage::new(TILE_DIM * 2, TILE_DIM * 2);
                image::imageops::overlay(&mut canvas, &processed_tiles[0], 0, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[1], TILE_DIM as i64, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[2], 0, TILE_DIM as i64);
                image::imageops::overlay(
                    &mut canvas,
                    &processed_tiles[3],
                    TILE_DIM as i64,
                    TILE_DIM as i64,
                );
                canvas
            }
            _ => continue,
        };

        let mut buf = Cursor::new(Vec::new());
        if final_image_buffer
            .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 75))
            .is_ok()
        {
            results.insert(preset.name.clone(), buf.into_inner());
        }
    }

    Ok(results)
}

#[tauri::command]
async fn save_temp_file(bytes: Vec<u8>) -> Result<String, String> {
    let mut temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    temp_file.write_all(&bytes).map_err(|e| e.to_string())?;
    let (_file, path) = temp_file.keep().map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
async fn save_panorama(
    first_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let panorama_image = state
        .panorama_result
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| {
            "No panorama image found in memory to save. It might have already been saved."
                .to_string()
        })?;

    let first_path = Path::new(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("panorama");

    let output_filename = format!("{}_Pano.png", stem);
    let output_path = parent_dir.join(output_filename);

    panorama_image
        .save(&output_path)
        .map_err(|e| format!("Failed to save panorama image: {}", e))?;

    Ok(output_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn load_and_parse_lut(
    path: String,
    state: tauri::State<'_, AppState>,
) -> Result<LutParseResult, String> {
    let lut = lut_processing::parse_lut_file(&path).map_err(|e| e.to_string())?;
    let lut_size = lut.size;

    let mut cache = state.lut_cache.lock().unwrap();
    cache.insert(path, Arc::new(lut));

    Ok(LutParseResult { size: lut_size })
}

fn apply_window_effect(theme: String, window: impl raw_window_handle::HasWindowHandle) {
    #[cfg(target_os = "windows")]
    {
        let color = match theme.as_str() {
            "light" => Some((250, 250, 250, 150)),
            "muted-green" => Some((44, 56, 54, 100)),
            _ => Some((26, 29, 27, 60)),
        };

        let info = os_info::get();

        let is_win11_or_newer = match info.version() {
            os_info::Version::Semantic(major, _, build) => *major == 10 && *build >= 22000,
            _ => false,
        };

        if is_win11_or_newer {
            window_vibrancy::apply_acrylic(&window, color)
                .expect("Failed to apply acrylic effect on Windows 11");
        } else {
            window_vibrancy::apply_blur(&window, color)
                .expect("Failed to apply blur effect on Windows 10 or older");
        }
    }

    #[cfg(target_os = "macos")]
    {
        let material = match theme.as_str() {
            "light" => window_vibrancy::NSVisualEffectMaterial::ContentBackground,
            _ => window_vibrancy::NSVisualEffectMaterial::HudWindow,
        };
        window_vibrancy::apply_vibrancy(&window, material, None, None)
            .expect("Unsupported platform! 'apply_vibrancy' is only supported on macOS");
    }

    #[cfg(target_os = "linux")]
    {}
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();

            let resource_path = app_handle
                .path()
                .resolve("resources", tauri::path::BaseDirectory::Resource)
                .expect("failed to resolve resource directory");

            let ort_library_name = {
                #[cfg(target_os = "windows")]
                {
                    "onnxruntime.dll"
                }
                #[cfg(target_os = "linux")]
                {
                    "libonnxruntime.so"
                }
                #[cfg(target_os = "macos")]
                {
                    "libonnxruntime.dylib"
                }
            };

            let ort_library_path = resource_path.join(ort_library_name);
            // TODO: Audit that the environment access only happens in single-threaded code.
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &ort_library_path) };
            println!("Set ORT_DYLIB_PATH to: {}", ort_library_path.display());

            let settings: AppSettings = load_settings(app_handle.clone()).unwrap_or_default();
            let window_cfg = app.config().app.windows.get(0).unwrap().clone();
            let transparent = settings.transparent.unwrap_or(window_cfg.transparent);
            let decorations = settings.decorations.unwrap_or(window_cfg.decorations);

            let window = tauri::WebviewWindowBuilder::from_config(app.handle(), &window_cfg)
                .unwrap()
                .transparent(transparent)
                .decorations(decorations)
                .build()
                .expect("Failed to build window");

            if transparent {
                let theme = settings.theme.unwrap_or("dark".to_string());
                apply_window_effect(theme, &window);
            }

            Ok(())
        })
        .manage(AppState {
            original_image: Mutex::new(None),
            cached_preview: Mutex::new(None),
            gpu_context: Mutex::new(None),
            gpu_image_cache: Mutex::new(None),
            export_task_handle: Mutex::new(None),
            panorama_result: Arc::new(Mutex::new(None)),
            indexing_task_handle: Mutex::new(None),
            lut_cache: Mutex::new(HashMap::new()),
            thumbnail_cancellation_token: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            load_image,
            apply_adjustments,
            export_image,
            batch_export_images,
            cancel_export,
            estimate_export_size,
            estimate_batch_export_size,
            generate_fullscreen_preview,
            generate_original_transformed_preview,
            generate_preset_preview,
            generate_uncropped_preview,
            generate_mask_overlay,
            update_window_effect,
            get_supported_file_types,
            stitch_panorama,
            save_panorama,
            load_and_parse_lut,
            fetch_community_presets,
            generate_all_community_previews,
            save_temp_file,
            image_processing::generate_histogram,
            image_processing::generate_waveform,
            image_processing::calculate_auto_adjustments,
            file_management::list_images_in_dir,
            file_management::get_folder_tree,
            file_management::generate_thumbnails,
            file_management::generate_thumbnails_progressive,
            cancel_thumbnail_generation,
            file_management::create_folder,
            file_management::delete_folder,
            file_management::copy_files,
            file_management::move_files,
            file_management::rename_folder,
            file_management::rename_files,
            file_management::duplicate_file,
            file_management::show_in_finder,
            file_management::delete_files_from_disk,
            file_management::delete_files_with_associated,
            file_management::save_metadata_and_update_thumbnail,
            file_management::apply_adjustments_to_paths,
            file_management::load_metadata,
            file_management::load_presets,
            file_management::save_presets,
            file_management::load_settings,
            file_management::save_settings,
            file_management::reset_adjustments_for_paths,
            file_management::apply_auto_adjustments_to_paths,
            file_management::handle_import_presets_from_file,
            file_management::handle_export_presets_to_file,
            file_management::save_community_preset,
            file_management::clear_all_sidecars,
            file_management::clear_thumbnail_cache,
            file_management::set_color_label_for_paths,
            file_management::import_files,
            tagging::start_background_indexing,
            tagging::clear_all_tags,
            culling::cull_images,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
