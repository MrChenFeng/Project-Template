import matplotlib.pyplot as plt


def display(image, label):
    t = plt.figure()
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.scatter(label[:, 0], label[:, 1])
    return t
