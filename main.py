from spam_detector import SpamDetector
from data import NEW_DATA, TRAINING_DATA, TRAINING_LABELS


def main():
    detector = SpamDetector()
    detector.train(TRAINING_DATA, TRAINING_LABELS)
    
    for index, email in enumerate(NEW_DATA):
        print(
            f"New email {index + 1}:" "\n"
            f"{email}" "\n"
            f"Result: {'is spam' if detector.is_spam(email) else 'not spam'}" "\n"
            "\n\n---\n\n"
        )

    input("Press Enter to exit")


if __name__ == "__main__":
    main()