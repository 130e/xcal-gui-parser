import tkinter as tk
from tkinter import ttk


def insert_treeview_entry(tree, piid, d):
    if isinstance(d, dict):
        for i, (k, v) in enumerate(d.items()):
            iid = piid + "-" + str(i)
            tree.insert(piid, tk.END, iid, text=k)
            insert_treeview_entry(tree, iid, v)


def show_treeview(data_dict, row_height=45):
    # create root window
    window = tk.Tk()
    window.title("Treeview - Hierarchical Data")
    window.geometry('600x900')

    # configure the grid layout
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    style = ttk.Style(window)
    style.configure("Treeview", rowheight=row_height)

    # create a treeview
    tree = ttk.Treeview(window)
    insert_treeview_entry(tree, "", data_dict)

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
