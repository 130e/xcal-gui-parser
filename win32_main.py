import pywinauto
import code
import subprocess

from fileread import XCALInfo
import fileview


def locate_window(keywords):
    keyWindow = app.top_window().child_window(best_match=keywords)
    editView = keyWindow.Edit
    listView = keyWindow.ListView
    return keyWindow, editView, listView


# need to run as admin

app = pywinauto.Application(backend="win32").connect(
    path=r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe"
)

keywords = ["TBaseForm", "5GNR NSA Status Information (Mobile1)TBaseForm"]
keyWindow, editView, listView = locate_window(keywords)

user_input = ""
while True:
    try:
        user_input = input("?>")
    except EOFError:
        print("Exiting")
        exit()

    if user_input == "reload":
        keyWindow, listView, editView = locate_window(keywords)
    elif user_input == "identifier":
        keyWindow.print_control_identifiers(filename="./identifiers.txt")
        subprocess.Popen("python.exe ./quick_view.py", shell=True)
    elif user_input == "cmd":
        code.interact(local=locals())
    elif user_input == "view":
        # parse and view
        text = editView.window_text()
        root = XCALInfo("root")
        root.add_children(
            [XCALInfo(line) for line in text.splitlines() if line.strip()]
        )
        d = root.as_dict()["root"]
        # print(d)
        fileview.show_treeview(d)
        print("Done")
    elif user_input == "dump":
        pass
    else:
        print("Unknown")
