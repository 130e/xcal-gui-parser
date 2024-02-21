import pywinauto
import code
import subprocess

from fileread import XCALMsgJSON, XCALMsgTable, MalformXCALMsgTable
from fileview import show_treeview


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


class XCALWindowControl:
    def __init__(
        self,
        file,
        keyword,
        formclass,
        listViewId,
    ) -> None:
        self.app = pywinauto.Application(backend="win32").connect(path=file)
        self.keyWindow = self.app.window(
            title_re="XCAL5.*", class_name="TMainForm"
        ).child_window(best_match=keyword, class_name=formclass)
        self.editView = self.keyWindow.child_window(
            best_match="Edit", class_name="TMemo"
        )
        self.listView = self.keyWindow.child_window(
            best_match="ListView" + listViewId, class_name="TJaheonListView"
        )
        self.reload()

    def reload(self):
        self.text = self.editView.window_text()

    def parse_next(self):
        self.listView.send_keystrokes("{DOWN}")
        self.reload()

    def process_text(self):
        # implement by subclass
        if self.text == None:
            return
        print(self.text)


class SignalMsg(XCALWindowControl):
    def __init__(
        self,
        file=r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe",
        keyword="Signalling Message",
        formclass="TfSignalMsgForm",
        listViewId="",
    ) -> None:
        self.app = pywinauto.Application(backend="win32").connect(path=file)
        self.keyWindow = self.app.window(
            title_re="XCAL5.*", class_name="TMainForm"
        ).child_window(best_match=keyword, class_name=formclass)
        self.editView = self.keyWindow.child_window(
            best_match="Edit", class_name="TMemo"
        )
        self.listView = self.keyWindow.child_window(
            best_match="ListView" + listViewId, class_name="TJaheonListView"
        )
        self.data = None
        self.reload()

    def reload(self):
        super().reload()
        if self.text == None:
            return
        root = XCALMsgJSON("root")
        root.add_children(
            [XCALMsgJSON(line) for line in self.text.splitlines() if line.strip()]
        )
        self.data = root.as_dict()

    def view(self):
        if self.data != None:
            show_treeview(self.data, 30)

    def find(self, key):
        if self.data != None:
            return [x for x in gen_dict_extract(key, self.data)]


class NSAStatusMsg(SignalMsg):
    def __init__(
        self,
        file=r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe",
        keyword="5GNR NSA Status Information (Mobile1)",
        formclass="TBaseForm",
        listViewId="",
    ) -> None:
        self.app = pywinauto.Application(backend="win32").connect(path=file)
        self.keyWindow = self.app.window(
            title_re="XCAL5.*", class_name="TMainForm"
        ).child_window(best_match=keyword, class_name=formclass)
        self.editView = self.keyWindow.child_window(
            best_match="Edit", class_name="TMemo"
        )
        self.listView = self.keyWindow.child_window(
            best_match="ListView" + listViewId, class_name="TJaheonListView"
        )


# work with DCI message and NON-malformed table
class TableTypeMsg(XCALWindowControl):
    def __init__(
        self,
        file=r"C:\Program Files (x86)\Accuver\XCAL5\XCAL5.exe",
        keyword="Qualcomm DM Message (Mobile1)",
        formclass="TBaseForm",
        listViewId="2",
    ) -> None:
        self.app = pywinauto.Application(backend="win32").connect(path=file)
        self.keyWindow = self.app.window(
            title_re="XCAL5.*", class_name="TMainForm"
        ).child_window(best_match=keyword, class_name=formclass)
        self.editView = self.keyWindow.child_window(
            best_match="Edit", class_name="TMemo"
        )
        self.listView = self.keyWindow.child_window(
            best_match="ListView" + listViewId, class_name="TJaheonListView"
        )
        self.reload()

    def reload(self):
        super().reload()
        if self.text == None:
            return
        self.data = XCALMsgTable(self.text).get_data()

class PDSCHMsg(TableTypeMsg):
    def reload(self):
        super().reload()
        if self.text == None:
            return
        self.data = MalformXCALMsgTable(self.text).get_data()


# need to run as admin
if __name__ == "__main__":
    xcal = SignalMsg()
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
        elif user_input == "nsa":
            # for reading NSA window
            xcal = NSAStatusMsg()
        elif user_input == "sig":
            xcal = SignalMsg()
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
