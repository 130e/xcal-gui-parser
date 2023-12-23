import pywinauto
import code
import subprocess

from fileread import XCALInfo
import fileview


def gen_dict_extract(key, var):
    if hasattr(var, "items"):
        for k, v in var.items():
            # if k == key:
            if key in k:
                if v != None:
                    yield v
                else:
                    yield k
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result


class XCALGrabber:
    def __init__(
        self, file, keyword="Qualcomm DM Message (Mobile1)", listViewId=""
    ) -> None:
        self.app = pywinauto.Application(backend="win32").connect(path=file)
        self.keyWindow = self.app.window(
            title_re="XCAL5.*", class_name="TMainForm"
        ).child_window(best_match=keyword, class_name="TBaseForm")
        self.editView = self.keyWindow.child_window(
            best_match="Edit", class_name="TMemo"
        )
        self.listView = self.keyWindow.child_window(
            best_match="ListView" + listViewId, class_name="TJaheonListView"
        )

    def reload(self):
        text = self.editView.window_text()
        if text == None:
            return
        self.text = text
        root = XCALInfo("root")
        root.add_children(
            [XCALInfo(line) for line in self.text.splitlines() if line.strip()]
        )
        self.data = root.as_dict()

    def view(self):
        fileview.show_treeview(self.data, 30)

    def find(self, key):
        return [x for x in gen_dict_extract(key, self.data)]

    def parse_next(self):
        self.listView.send_keystrokes("{DOWN}")
        self.reload()


# need to run as admin

if __name__ == "__main__":
    xcal = XCALGrabber(
        r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe", listViewId="2"
    )
    print("Default window loaded.")

    user_input = ""
    while True:
        try:
            user_input = input("?>")
        except EOFError:
            print("Exiting")
            exit()

        if user_input == "id":
            xcal.keyWindow.print_control_identifiers(filename="./identifiers.txt")
            # subprocess.Popen("python.exe ./", shell=True)
        elif user_input == "new":
            # for reading NSA window for now
            # keyword = input("New window keyword")
            keyword = "5GNR NSA Status Information (Mobile1)"
            xcal = XCALGrabber(
                r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe",
                keyword=keyword,
                listViewId="",
            )
        elif user_input == "cmd":
            code.interact(local=locals())
        elif user_input == "view":
            # parse and view
            xcal.reload()
            xcal.view()
            print("Done")
        elif user_input == "find":
            key = input("Keyword")
            xcal.reload()
            result = xcal.find(key)
            print(len(result), "occurences found")
            print(result)
        elif user_input == "next":
            xcal.parse_next()
            xcal.view()
        elif user_input == "dump":
            # such as Header_E1 : 1
            # TODO: allow regex match, such as Header_E1 : [1-4]
            # TODO: hit stop button when found
            key = input("The keyword to find in all:")
            xcal.reload()
            while True:
                result = xcal.find(key)
                if len(result) > 0:
                    print(result)
                    break
                xcal.parse_next()
        else:
            print("Unknown")
