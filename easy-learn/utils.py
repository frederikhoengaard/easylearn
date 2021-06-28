

class Bunch(dict):
    def __init__(self):
        super().__init__(self)

    def __getattr__(self, key):
        return self[key]


def main():
    pass
    
if __name__ == '__main__':
    main()