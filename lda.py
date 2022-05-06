import os
import math
import jieba
import numpy as np
from gensim import corpora, models
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(path, ban_stop_words=False, stop_words_path=''):
    data = []
    names = []
    stop_words = set()
    stop_txt = os.listdir(stop_words_path)
    for file in stop_txt:
        with open(stop_words_path + '/' + file, 'r', encoding='ANSI') as f:
            for j in f.readlines():
                stop_words.add(j.strip('\n'))
    replace = '[a-zA-Z0-9’!"#$%&\'()（）；：‘“？、》。《，*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+\n\u3000 '
    files = os.listdir(path)
    for file in files:
        with open(path + '/' + file, 'r', encoding='ANSI') as f:
            t = f.read()
            for i in replace:
                t = t.replace(i, '')
            if ban_stop_words:
                for i in stop_words:
                    t = t.replace(i, '')
            c = jieba.lcut(t)
            data.append(c)
        f.close()
        print("{} loaded".format(file))
        names.append(file.split(".txt")[0])
    return data, names


if __name__ == '__main__':
    words = 2000
    topics = 100
    train_paragraphs = 2000
    test_paragraphs = math.ceil(0.2*train_paragraphs)
    ban_stop_words = True
    data, text_names = load_data("./data", ban_stop_words, "./stop")
    text_num = len(data)

    train_data = []
    train_label = []
    for i in range(text_num):
        for j in range(math.ceil(train_paragraphs/text_num)):
            start = np.random.randint(0, len(data[i])-words-1)
            train_data.append(data[i][start:start+words])
            train_label.append(i)

    test_data = []
    test_label = []
    for i in range(text_num):
        for j in range(math.ceil(test_paragraphs/text_num)):
            start = np.random.randint(0, len(data[i])-words-1)
            test_data.append(data[i][start:start+words])
            test_label.append(i)

    dictionary = corpora.Dictionary(train_data)
    train_corpus = [dictionary.doc2bow(t) for t in train_data]
    lda = models.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=topics)
    train_distribution = lda.get_document_topics(train_corpus)
    train_matrix = np.zeros((len(train_label), topics))
    for i in range(len(train_distribution)):
        for j in train_distribution[i]:
            train_matrix[i][j[0]] = j[1]

    test_corpus = [dictionary.doc2bow(t) for t in test_data]
    test_distribution = lda.get_document_topics(test_corpus)
    test_matrix = np.zeros((len(test_label), topics))
    for i in range(len(test_distribution)):
        for j in test_distribution[i]:
            test_matrix[i][j[0]] = j[1]

    net_x_train = torch.FloatTensor(train_matrix)
    net_y_train = torch.LongTensor(train_label)
    net_x_test = torch.FloatTensor(test_matrix)
    net_y_test = torch.LongTensor(test_label)
    train_loader = DataLoader(TensorDataset(net_x_train, net_y_train), batch_size=96, shuffle=True)
    net = torch.nn.Sequential(
        torch.nn.Linear(topics, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, len(text_names))
    )
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=4e-2, params=net.parameters())

    for epoch in range(500):
        net.train()
        sum_loss = 0
        sum_acc = 0

        for batch_x, batch_y in train_loader:
            batch_y_pred = net(batch_x)
            batch_y_pred_label = torch.argmax(batch_y_pred, dim=1)
            sum_acc += torch.eq(batch_y_pred_label, batch_y).float().sum()
            loss = loss_func(batch_y_pred, batch_y)
            sum_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print("Epoch: %d/500 || Train || sum loss: %.4f || train_acc: %.4f || %d/%d"
                  % (epoch+1, sum_loss, sum_acc/len(train_label), sum_acc, len(train_label)))

    net.eval()
    pre = net(net_x_test)
    predictions = torch.argmax(pre, dim=1)
    acc = torch.eq(predictions, net_y_test).float().sum()
    loss = loss_func(pre, net_y_test)
    print("Test || loss: %.4f || test_acc: %.4f || %d/%d" % (loss, acc/len(test_label), acc, len(test_label)))
