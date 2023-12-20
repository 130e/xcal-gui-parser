import time
import pywinauto
import code
import subprocess

import fileread
import fileview


def locate_window(label):
    dlg_main = app.top_window().child_window(title="Workspace", control_type="Pane")
    dlg = dlg_main.child_window(title=label, control_type="Window", visible_only=False)
    # data panels
    listView = dlg.child_window(best_match="ListBox", visible_only=False)
    editView = dlg.child_window(best_match="Edit", visible_only=False)
    return dlg_main, dlg, listView, editView


# need to run as admin

app = pywinauto.Application(backend="win32").connect(
    path=r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe"
)

label = "5GNR NSA Status Information (Mobile1)"
dlg_main, dlg, listView, editView = locate_window(label)

user_input = ""
while True:
    try:
        user_input = input("?>")
    except EOFError:
        print("Exiting")
        exit()

    if user_input == "reload":
        dlg_main, dlg, listView, editView = locate_window(label)
    elif user_input == "view id":
        dlg.print_control_identifiers(filename="./identifiers.txt")
        subprocess.Popen("python.exe ./quick_view.py", shell=True)
    elif user_input == "cmd":
        code.interact(local=locals())
    elif user_input == "view":
        # parse and view
        print(editView.get_value())
    else:
        print()