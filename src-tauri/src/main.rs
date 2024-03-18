// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::GlobalShortcutManager;
use tauri::Manager;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let main_window = app.get_window("main").unwrap();
            main_window.set_decorations(false)?;
            app.global_shortcut_manager()
                .register("CmdOrControl+Alt+Shift+A", move || {
                    if let Ok(visible) = main_window.is_visible() {
                        if visible {
                            main_window.hide().unwrap();
                        } else {
                            main_window.show().unwrap();
                            main_window.set_focus().unwrap();
                        }
                    }
                })?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
