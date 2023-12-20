import tkinter as tk
from tkinter import ttk


def insert_treeview_entry(tree, piid, iid, d):
    if isinstance(d, dict):
        key = next(iter(d))
        tree.insert(piid, tk.END, iid, text=key)
        children = d[key]
        if not isinstance(children, list):
            children = [children]
        for i, entry in enumerate(children):
            insert_treeview_entry(tree, iid, iid + "-" + str(i), entry)
    elif isinstance(d, str):
        tree.insert(piid, tk.END, iid, text=d)
    else:
        print("Error: entry is not dict or text")


def show_treeview(data_dict, row_height=45):
    # create root window
    window = tk.Tk()
    window.title("Treeview - Hierarchical Data")
    # window.geometry('400x200')

    # configure the grid layout
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    style = ttk.Style(window)
    style.configure("Treeview", rowheight=row_height)

    # create a treeview
    tree = ttk.Treeview(window)
    if not isinstance(data_dict, list):
        data_dict = [data_dict]
    for i, entry in enumerate(data_dict):
        insert_treeview_entry(tree, "", str(i), entry)

    # place the Treeview widget on the root window
    tree.grid(row=0, column=0, sticky=tk.NSEW)
    # run the app
    window.mainloop()


# if __name__ == "__main__":
#     # create root window
#     window = tk.Tk()
#     window.title('Treeview - Hierarchical Data')
#     # window.geometry('400x200')

#     # configure the grid layout
#     window.rowconfigure(0, weight=1)
#     window.columnconfigure(0, weight=1)
#     style = ttk.Style(window)
#     style.configure('Treeview', rowheight=45)

#     # create a treeview
#     tree = ttk.Treeview(window)
#     # get data from file
#     with open('./identifiers.txt', 'r') as f:
#         f.readline()
#         f.readline()

#         itree = InfoTree(f.readlines(), '|')
#         data_dict = itree.dict()

#         insert_treeview_entry(tree, '', '0', data_dict)

#         # place the Treeview widget on the root window
#         tree.grid(row=0, column=0, sticky=tk.NSEW)

#         # run the app
#         window.mainloop()
