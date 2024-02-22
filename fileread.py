import re
import pandas as pd


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
            elif node.level <= self.level:
                # this node is a sibling, no more children
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


# Typically used in RRC and SIB, signal types messages in JSON like format
# need specify indentation
class XCALMsgJSON(Node):
    def __init__(self, indented_line, nspace=3):
        self.children = {}
        self.level = (len(indented_line) - len(indented_line.lstrip())) / nspace
        self.text = indented_line.strip()


# doesn't work for PDSCH status
# for 5G MAC DCI
class XCALMsgTable:
    def __init__(self, text) -> None:
        self.data = []
        data = text.splitlines()

        # find message table
        separator_ind = []
        for i, line in enumerate(data):
            if re.match(r"-{5,}", line):
                separator_ind.append(i)
                # print(i)
        if len(separator_ind) % 3 != 0:
            print("Error reading")

        # process table
        # check num of table
        i = 0
        separators = []
        while i + 2 <= len(separator_ind):
            separators.append(
                [separator_ind[i], separator_ind[i + 1], separator_ind[i + 2]]
            )
            i += 3

        # read each table
        for sep in separators:
            # parse table
            self.data.append(
                self.parse_table(data[sep[0] + 1 : sep[1]], data[sep[1] + 1 : sep[2]])
            )

    def parse_table_header(self, header_lines):
        pattern = r"\s{2,}(\w+)(\s(\w+)){0,}"
        loc2word = {0:"Table_index"}
        # pattern = r"(\w+)"
        # loc2word = {}

        # XCAL table text is right edge aligned
        # we use it to find out column names
        matches = re.finditer(pattern, header_lines[-1])
        for each in matches:
            loc2word[each.end()] = each.group()

        for line in reversed(header_lines[:-1]):
            matches = re.finditer(pattern, line)
            for each in matches:
                loc2word[each.end()] = "{} {}".format(
                    each.group(), loc2word[each.end()]
                )

        # change all empty space to "_"
        # due to C_RNTI and RA_RNTI container has differernt format
        column_names = [loc2word[k].replace("_", " ") for k in sorted(loc2word.keys())]
        
        return column_names

    def parse_table(self, header_lines, value_lines):
        # parse header
        column_names = self.parse_table_header(header_lines)
        # parse value
        column_values = []
        N = len(column_names)
        for line in value_lines:
            values = line.strip().split()
            # The misalignment is caused by empty slot
            # So the first column, table index is aligned
            if len(values) < N:
                values = values[0] + [None] * (N - len(values)) + values[1:]
            column_values.append(values)

        # df
        df = pd.DataFrame(column_values, columns=column_names)
        # remove dup col TODO: rename it
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df

    def get_data(self):
        return self.data

    # def column(self, column_name, table_index=0):
    #     tab = self.data[table_index]
    #     return tab[column_name].to_list()

# I hate XCAL
class MalformXCALMsgTable(XCALMsgTable):
    def parse_table_header(self, header_lines):
        pattern = r"\s{2,}(\w+)(\s(\w+)){0,}"
        loc2word = {0:"Table_index"}
        # XCAL table text is right edge aligned
        # we use it to find out column names
        matches = re.finditer(pattern, header_lines[-1])
        for each in matches:
            # print(each.group(), each.end())
            loc2word[each.end()] = ' '.join(each.group().split())

        for line in reversed(header_lines[:-1]):
            matches = re.finditer(pattern, line)
            for each in matches:
                # print(each.group(), each.end())
                loc2word[each.end()] = "{} {}".format(
                    ' '.join(each.group().split()), loc2word[each.end()]
                )

        # change all empty space to "_"
        # due to C_RNTI and RA_RNTI container has differernt format
        column_names = [loc2word[k].replace("_", " ") for k in sorted(loc2word.keys())]

        # special case
        for i,c in enumerate(column_names):
            if c == "Rx":
                if column_names[i-1] == "Num":
                    column_names[i-1] = "Num Rx"
                    column_names.pop(i)
                    break
        
        return column_names

if __name__ == "__main__":
    with open("./result.txt", "r") as f:
        text = f.read()
        root = XCALMsgJSON("root")
        root.add_children(
            [XCALMsgJSON(line) for line in text.splitlines() if line.strip()]
        )
        d = root.as_dict()
        print(d)
# if __name__ == "__main__":
#     with open("./result.txt", "r") as f:
#         itree = InfoTree(f.readlines(), "   ")
#         d = itree.dict()
#         print(d)
