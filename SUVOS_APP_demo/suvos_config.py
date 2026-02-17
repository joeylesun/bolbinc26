import tkinter as tk
from tkinter import messagebox
from suvos_common import load_fsm_config, save_fsm_config


def run_config_editor(parent_root=None):
    # Create a Toplevel window (popup) linked to the main launcher
    win = tk.Toplevel(parent_root)
    win.title("SUVOS Timer Configuration")
    win.geometry("350x300")

    # Load current values
    current_cfg = load_fsm_config()
    entries = {}

    tk.Label(win, text="Edit FSM Timers", font=("Arial", 14, "bold")).pack(pady=15)

    def add_field(label_text, key):
        frame = tk.Frame(win)
        frame.pack(pady=5, fill='x', padx=30)

        lbl = tk.Label(frame, text=label_text, width=20, anchor='w')
        lbl.pack(side='left')

        entry = tk.Entry(frame)
        entry.insert(0, str(current_cfg.get(key, 0.0)))
        entry.pack(side='right', expand=True, fill='x')

        entries[key] = entry

    # Create Fields
    add_field("Wait Time (sec):", "waitingTime")
    add_field("Clean Time (sec):", "disinfectionTime")
    add_field("Standby Time (sec):", "standbyTime")

    def save():
        try:
            new_cfg = {
                "waitingTime": float(entries["waitingTime"].get()),
                "disinfectionTime": float(entries["disinfectionTime"].get()),
                "standbyTime": float(entries["standbyTime"].get())
            }
            if save_fsm_config(new_cfg):
                messagebox.showinfo("Success", "Timers updated successfully!")
                win.destroy()  # Close popup
            else:
                messagebox.showerror("Error", "Failed to save file.")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    tk.Button(win, text="Save & Close", command=save, bg="#4CAF50", height=2, width=20).pack(pady=30)