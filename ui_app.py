import tkinter as tk
import subprocess

process = None

def start_monitoring():
    global process
    status_label.config(text="Status: Monitoring...", fg="#00cc66")
    process = subprocess.Popen(["python", "eye_strain_detector.py"])

def stop_monitoring():
    global process
    if process:
        process.terminate()
        status_label.config(text="Status: Stopped", fg="red")

# Window
root = tk.Tk()
root.title("EyeGuard AI")
root.geometry("500x350")
root.configure(bg="#0f172a")

# Title
title = tk.Label(
    root,
    text="EyeGuard AI",
    font=("Segoe UI", 28, "bold"),
    fg="white",
    bg="#0f172a"
)
title.pack(pady=20)

# Subtitle
subtitle = tk.Label(
    root,
    text="Eye Strain Monitoring System",
    font=("Segoe UI", 12),
    fg="#cbd5f5",
    bg="#0f172a"
)
subtitle.pack()

# Button Frame
button_frame = tk.Frame(root, bg="#0f172a")
button_frame.pack(pady=40)

# Start Button
start_btn = tk.Button(
    button_frame,
    text="Start Monitoring",
    font=("Segoe UI", 12, "bold"),
    bg="#22c55e",
    fg="white",
    padx=20,
    pady=10,
    borderwidth=0,
    command=start_monitoring
)
start_btn.grid(row=0, column=0, padx=10)

# Stop Button
stop_btn = tk.Button(
    button_frame,
    text="Stop Monitoring",
    font=("Segoe UI", 12, "bold"),
    bg="#ef4444",
    fg="white",
    padx=20,
    pady=10,
    borderwidth=0,
    command=stop_monitoring
)
stop_btn.grid(row=0, column=1, padx=10)

# Status Label
status_label = tk.Label(
    root,
    text="Status: Not Running",
    font=("Segoe UI", 11),
    fg="#facc15",
    bg="#0f172a"
)
status_label.pack(pady=10)

# Footer
footer = tk.Label(
    root,
    text="AI Powered Eye Monitoring",
    font=("Segoe UI", 9),
    fg="#94a3b8",
    bg="#0f172a"
)
footer.pack(side="bottom", pady=10)

root.mainloop()