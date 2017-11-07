
def cardAt(n):
    k = ['C', 'D', 'H', 'S']
    l = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
    result=[]
    for i in k:
        for j in l:
            result.append(j+i)
    try:
        print(result[n])
    except IndexError:
        print("Not have Card")

if __name__ == '__main__':
    cardAt(0)
    cardAt(1)
    cardAt(34)
    cardAt(35)
    cardAt(10)