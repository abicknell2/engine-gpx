#! /usr/local/bin/python

# return a set of selected values when a string in the form:
# 1-4,6
# would return:
# 1,2,3,4,6
# as expected...


def parseIntSet(nputstr: str = "") -> set[int]:
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(",")]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = f"1-{i[1:]}"
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except BaseException:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split("-")]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except BaseException:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        # TODO: raise an exception
        print("Invalid set: ", str(invalid))
    return selection


# end parseIntSet


def parseIntList(nputstr: str = "") -> list[int]:
    selection = []
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(",")]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = f"1-{i[1:]}"
        try:
            # typically tokens are plain old integers
            selection.append(int(i))
        except BaseException:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split("-")]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    selection.extend(range(first, last + 1))
            except BaseException:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        # TODO: rasie an exception
        print("Invalid set: ", str(invalid))
    return selection


# end parseIntSet


def parseIntListGaps(nputstr: str = "", source: list[int] = []) -> list[int]:
    """Parse ints into a list with gaps based on a sorted list of possible values.

    Arguments
    ---------
    source : list
        the numbers which should be drawn from
    nputstr : string
        the string to parse into the indicies


    Returns
    -------
    list of resulting indicies

        Typical usage example:

        source_list = [100, 120, 130, 135, 140, 150, 160]
        input_string = '100-140, 160'
        result --> [100, 120, 130, 135, 140, 160]

    """

    if nputstr.replace(" ", "") == "" or len(source) == 0:
        return []
    # get the full list
    full_list = parseIntList(nputstr=nputstr)
    # remove entries which are not in the source list
    return [val for val in full_list if val in source]


# end parseIntListGaps

if __name__ == "__main__":

    print("Generate a list of selected items!")
    nputstr = input("Enter a list of items: ")

    selection = parseIntList(nputstr)
    print("Your selection is:\n", selection)
