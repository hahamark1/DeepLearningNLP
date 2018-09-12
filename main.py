from src.DataLoader import DataLoader
from src.PreProcessor import PreProcessor

def main():
    DL = DataLoader()
    DL.load_train_comments()
    DL.load_test_comments()
    DL.load_vocabulaire()
    DL.load_word_sentiment()
    DL.load_twitter_comments()

    PP = PreProcessor(DL, stemming=True, stopwords_removal=True, charecter_removal=True)


if __name__ == '__main__':
    main()
