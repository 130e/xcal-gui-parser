class Node:
    def __init__(self, indented_line):
        self.children = {}
        self.level = len(indented_line) - len(indented_line.lstrip())
        self.text = indented_line.strip()

    def add_children(self, node_lists):
        childlevel = node_lists[0].level
        last_child = None
        while node_lists:
            node = node_lists.pop(0)
            if node.level == childlevel:  # add node as a child
                self.children[node.text] = node
                last_child = node
            elif (
                node.level > childlevel
            ):  # add nodes as grandchildren of the last child
                node_lists.insert(0, node)
                last_child.add_children(node_lists)
            elif node.level <= self.level:  # this node is a sibling, no more children
                node_lists.insert(0, node)
                return

    def as_dict(self):
        if len(self.children) >= 1:
            ret = dict()
            for k, v in self.children.items():
                ret[k] = v.as_dict()
            return ret
        else:
            return None


class Identifier(Node):
    def __init__(self, text, level):
        self.children = {}
        self.level = level
        self.text = text


def level_parsing(text, sep):
    level = 0
    prev_length = len(text)
    while True:
        text = text.lstrip(sep).lstrip()
        if len(text) == prev_length:
            break
        level += 1
        prev_length = len(text)
    return level, text


def parse_identifiers(lines, sep="|"):
    root = Identifier("root", 0)
    items = []
    entry = str()
    cur_level = 0
    content = ""
    for line in lines:
        level, content = level_parsing(line, sep)
        if content != "":  # not empty
            entry += content
        elif entry != "":  # empty row
            # print(cur_level, text)
            items.append(Identifier(entry, cur_level))
            entry = str()
        cur_level = level
        # input()
    if content != "":  # add last entry
        items.append(Identifier(entry, cur_level))
    root.add_children(items)
    return root.as_dict()


class XCALInfo(Node):
    def __init__(self, indented_line, nspace=3):
        self.children = {}
        self.level = (len(indented_line) - len(indented_line.lstrip())) / nspace
        self.text = indented_line.strip()


if __name__ == "__main__":
    with open("./result.txt", "r") as f:
        text = f.read()
        root = XCALInfo("root")
        root.add_children(
            [XCALInfo(line) for line in text.splitlines() if line.strip()]
        )
        d = root.as_dict()
        print(d)
# if __name__ == "__main__":
#     with open("./result.txt", "r") as f:
#         itree = InfoTree(f.readlines(), "   ")
#         d = itree.dict()
#         print(d)
