use anyhow::Result;
use futures::stream::{self, StreamExt};
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::task::JoinHandle;
use walkdir::WalkDir;

use crate::AppState;
use crate::file_management::{self, get_sidecar_path};
use crate::formats::is_supported_image_file;
use crate::image_processing::ImageMetadata;

pub const COLOR_TAG_PREFIX: &str = "color:";

#[tauri::command]
pub async fn start_background_indexing(
    folder_path: String,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if let Some(handle) = state.indexing_task_handle.lock().unwrap().take() {
        println!("Cancelling previous indexing task.");
        handle.abort();
    }

    let settings = file_management::load_settings(app_handle.clone())?;
    if !settings.enable_ai_tagging.unwrap_or(false) {
        return Ok(());
    }

    let max_concurrent_tasks = settings.tagging_thread_count.unwrap_or(3).max(1) as usize;

    let app_handle_clone = app_handle.clone();

    let task: JoinHandle<()> = tokio::spawn(async move {
        let _ = app_handle_clone.emit("indexing-started", ());
        println!("Starting background indexing for: {}", folder_path);
        println!(
            "Using {} concurrent threads for AI tagging.",
            max_concurrent_tasks
        );

        let state_clone = app_handle_clone.state::<AppState>();
        let gpu_context = crate::gpu_processing::get_or_init_gpu_context(&state_clone).ok();

        let image_paths: Vec<PathBuf> = match fs::read_dir(&folder_path) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file() && is_supported_image_file(&path.to_string_lossy()))
                .collect(),
            Err(e) => {
                eprintln!("Failed to read directory '{}': {}", folder_path, e);
                let _ = app_handle_clone
                    .emit("indexing-error", format!("Failed to read directory: {}", e));
                *app_handle_clone
                    .state::<AppState>()
                    .indexing_task_handle
                    .lock()
                    .unwrap() = None;
                return;
            }
        };

        println!(
            "Found {} images to process in {}",
            image_paths.len(),
            folder_path
        );
        let total_images = image_paths.len();
        let processed_count = Arc::new(Mutex::new(0));

        stream::iter(image_paths)
            .for_each_concurrent(max_concurrent_tasks, |path| {
                let app_handle_inner = app_handle_clone.clone();
                let gpu_context_inner = gpu_context.clone();
                let processed_count_inner = Arc::clone(&processed_count);

                async move {
                    let path_str = path.to_string_lossy().to_string();
                    let sidecar_path = get_sidecar_path(&path_str);

                    let metadata: ImageMetadata = if sidecar_path.exists() {
                        fs::read_to_string(&sidecar_path)
                            .ok()
                            .and_then(|c| serde_json::from_str(&c).ok())
                            .unwrap_or_default()
                    } else {
                        ImageMetadata::default()
                    };

                    if metadata.tags.is_none() {
                        let _ = file_management::get_cached_or_generate_thumbnail_image(
                            &path_str,
                            &app_handle_inner,
                            gpu_context_inner.as_ref(),
                        );
                    }

                    let mut count = processed_count_inner.lock().unwrap();
                    *count += 1;
                    let _ = app_handle_inner.emit(
                        "indexing-progress",
                        serde_json::json!({
                            "current": *count,
                            "total": total_images
                        }),
                    );
                }
            })
            .await;

        println!("Background indexing finished for: {}", folder_path);
        let _ = app_handle_clone.emit("indexing-finished", ());

        *app_handle_clone
            .state::<AppState>()
            .indexing_task_handle
            .lock()
            .unwrap() = None;
    });

    *state.indexing_task_handle.lock().unwrap() = Some(task);

    Ok(())
}

#[tauri::command]
pub fn clear_all_tags(root_path: String) -> Result<usize, String> {
    if !Path::new(&root_path).exists() {
        return Err(format!("Root path does not exist: {}", root_path));
    }

    let mut updated_count = 0;
    let walker = WalkDir::new(root_path).into_iter();

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rrdata") {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(mut metadata) = serde_json::from_str::<ImageMetadata>(&content) {
                    if let Some(tags) = &mut metadata.tags {
                        let original_len = tags.len();
                        tags.retain(|tag| tag.starts_with(COLOR_TAG_PREFIX)); // don't remove color tags, just AI tags

                        if tags.len() < original_len {
                            if tags.is_empty() {
                                metadata.tags = None;
                            }
                            if let Ok(json_string) = serde_json::to_string_pretty(&metadata) {
                                if fs::write(path, json_string).is_ok() {
                                    updated_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(updated_count)
}
