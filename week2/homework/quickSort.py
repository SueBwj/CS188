
def quickSortHelper(list, left, right):
    if left >= right:
        return
    else:
        i, j = left, right
        while i < j:
            while i < j and list[j] >= list[left]:
                j -= 1
            while i < j and list[i] <= list[left]:
                i += 1
            list[i], list[j] = list[j], list[i]
        list[i], list[left] = list[left], list[i]
        quickSortHelper(list, 0, i-1)
        quickSortHelper(list, i+1, right)


def quickSort(list):
    # 如果列表长度小于等于1则直接返回
    if len(list) <= 1:
        return list
    # 调用Helper函数解决问题
    quickSortHelper(list, 0, len(list)-1)
    return list


if __name__ == '__main__':
    lst = [2, 4, 5, 1]
    print(quickSort(lst))
