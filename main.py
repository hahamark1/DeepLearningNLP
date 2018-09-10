from src.DataLoader import DataLoader

def main():
    DL = DataLoader()
    DL.load_train_comments()
    DL.load_test_comments()
    DL.load_vocabulaire()

if __name__ == '__main__':
    main()
