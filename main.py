from src.DataLoader import DataLoader
from src.Preprocessor import PreProcessor

def main():
    DL = DataLoader()
    batch = next(iter(DL.train_loader))
    print(len(batch.text))
    print(batch.label)
    print())
    # PP = PreProcessor(DL, stemming=True, stopwords_removal=True, charecter_removal=True)

if __name__ == '__main__':
    main()
