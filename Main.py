# coding=utf-8

import MnistLoader as loader
import NetMLP as network

def main():
    training_data, validation_data, test_data = loader.load_data_wrapper()
    net = network.NetMLP([784, 30, 10])
    net.SDG(training_data, 30, 10, 0.01, test_data = test_data)

if __name__ == '__main__':
    main()
