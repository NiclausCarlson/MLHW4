import Letter
import Utils


def getNgram(arr, n):
    return list(zip(*[arr[i:] for i in range(n)]))


# частоный анализ (проводится не правильно, надо ещё разибить на спам/не спам)
def generateCountMap(ngrams):
    nmaps = {}
    for ngram in ngrams:  # каждая n-грама - это Letter, обработанный как n-грамма
        tmpMap = {}
        tmpList = ngram.subject + ngram.letter
        for gramma in tmpList:
            if gramma not in tmpMap:
                if gramma in nmaps:
                    nmaps[gramma] += 1
                else:
                    nmaps[gramma] = 0
    return nmaps


class Bayes:
    def __init__(self):
        self.MAX_NGRAM_LENGTH = 3
        self.MIN_ALPHA_PARAMETER = 1e-3
        self.MAX_ALPHA_PARAMETER = 1e-2
        self.ALPHA_STEP = 1e-3
        self.DATASETS = Utils.getParts()
        self.BLOCK_QUANTITY = 10
        self.Q = 2  # spam or not to spam

    def createNgrams(self, n, indexes):
        ngragms = []  # делаем n-грамы для каждого письма
        for i in indexes:  # перебираем индексы обучающих блоков
            for letter in self.DATASETS[i]:
                subject = getNgram(letter.subject, n)
                text = getNgram(letter.letter, n)
                ngragms.append(Letter.Letter(subject, text, letter.isSpam))
        return ngragms

    def bayes(self):
        # будем проводить k-fold валидацию для каждого значения n и alpha
        for curNgramLength in range(self.MAX_NGRAM_LENGTH):
            alpha = self.MIN_ALPHA_PARAMETER
            while alpha <= self.MIN_ALPHA_PARAMETER:
                for testDataIndex in range(self.BLOCK_QUANTITY):
                    indexes = [i for i in range(self.BLOCK_QUANTITY) if i != testDataIndex]
                    nGrams = self.createNgrams(curNgramLength + 1, indexes)
                    countMaps = generateCountMap(nGrams)
                    # считаем вероятности

                alpha += self.ALPHA_STEP

    def findLambdas(self):
        return 0


b = Bayes()
b.bayes()
