from network import *
from tqdm import tqdm
import argparse
import pickle
import gzip

def extract_data(filename, num_images, IMAGE_WIDTH):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

parser = argparse.ArgumentParser()
parser.add_argument('weights')
parser.add_argument('test_images')
parser.add_argument('test_labels')

if __name__ == '__main__':
    args = parser.parse_args()
    params = pickle.load(open(args.weights, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Get test data
    m =10000
    X = extract_data(args.test_images, m, 28)
    y_dash = extract_labels(args.test_labels, m).reshape(m,1)

    # Normalize the data
    X/= 255
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1] #from index 0 till second to last index (pixel information)
    X = X.reshape(len(test_data), 28, 28,1)
    y = test_data[:,-1] #last index (lable information)

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x,i,f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))

    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))