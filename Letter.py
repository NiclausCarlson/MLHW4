class Letter:
    def __init__(self, subject, letter, isSpam):
        self.subject = subject
        self.letter = letter
        self.isSpam = isSpam

    def print(self):
        print("Subject: " + str(self.subject))
        print("Letter: " + str(self.letter))
        print("IsSpam: " + str(self.isSpam))
