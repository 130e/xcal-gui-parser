class Node:
    def __init__(self, indented_line):
        self.children = []
        self.level = len(indented_line) - len(indented_line.lstrip())
        self.text = indented_line.strip()

    def add_children(self, nodes):
        childlevel = nodes[0].level
        while nodes:
            node = nodes.pop(0)
            if node.level == childlevel:  # add node as a child
                self.children.append(node)
            elif (
                node.level > childlevel
            ):  # add nodes as grandchildren of the last child
                nodes.insert(0, node)
                self.children[-1].add_children(nodes)
            elif node.level <= self.level:  # this node is a sibling, no more children
                nodes.insert(0, node)
                return

    def as_dict(self):
        if len(self.children) > 1:
            return {self.text: [node.as_dict() for node in self.children]}
        elif len(self.children) == 1:
            return {self.text: self.children[0].as_dict()}
        else:
            return self.text


class Identifier(Node):
    def __init__(self, text, level):
        self.children = []
        self.level = level
        self.text = text


class IdentifierTree:
    def __init__(self, lines, sep):
        self.root = Identifier("root", 0)
        items = []
        text = str()
        cur_level = 0
        content = ""
        for line in lines:
            level, content = self.__level_parsing(line, sep)
            if content != "":  # not empty
                text += content
            elif text != "":
                print(cur_level, text)
                items.append(Identifier(text, cur_level))
                text = str()
            cur_level = level
            input()
        if content != "":
            items.append(Identifier(text, cur_level))
        self.root.add_children(items)

    def __level_parsing(text, sep):
        level = 0
        prev_length = len(text)
        while True:
            text = text.lstrip(sep).lstrip()
            if len(text) == prev_length:
                break
            level += 1
            prev_length = len(text)
        return level, text

    def dict(self):
        return self.root.as_dict()["root"]


class XCALInfo(Node):
    def __init__(self, indented_line, nspace=3):
        self.children = []
        self.level = (len(indented_line) - len(indented_line.lstrip())) / nspace
        self.text = indented_line.strip()


if __name__ == "__main__":
    with open("./result.txt", "r") as f:
        text = f.read()
        root = XCALInfo("root")
        root.add_children(
            [XCALInfo(line) for line in text.splitlines() if line.strip()]
        )
        d = root.as_dict()["root"]
        print(d)

# if __name__ == "__main__":
#     with open("./result.txt", "r") as f:
#         itree = InfoTree(f.readlines(), "   ")
#         d = itree.dict()
#         print(d)
