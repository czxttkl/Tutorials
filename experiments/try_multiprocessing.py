# running under python 3.4 / Windows
# but behaves the same under Unix
import multiprocessing as mp

x = 0


class A:
    y = 0


def f():
    print("f.x", x)
    print("f.A.y", A.y)


def g(xx, A):
    global x
    print("g.x", x)
    print("g.xx", xx)
    print("g.A.y", A.y)


def main():
    global x
    x = 1
    A.y = 1
    p = mp.Process(target=f)
    p.start()
    q = mp.Process(target=g, args=(x, A))
    q.start()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
