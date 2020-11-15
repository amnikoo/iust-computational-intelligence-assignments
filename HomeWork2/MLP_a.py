# *-* coding: utf-8 *-*

from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb

# type(train_images):  <class 'list'>
# len(train_images):  60000
#
# type(train_images[ i ]): <class 'numpy.ndarray'>
# train_images[ i ].dtype: uint8
# train_images[ i ].min(): 0
# train_images[ i ].max(): 255
# train_images[ i ].shape: (HEIGHT, WIDTH)
#
# type(train_labels):  <class 'list'>
# len(train_labels):  60000
#
# type(train_labels[ i ]): <class 'int'>
# train_labels[ i ]: 0...9
print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')


# type(test_images):  <class 'list'>
# len(test_images):  20000
#
# type(test_images[ i ]): <class 'numpy.ndarray'>
# test_images[ i ].dtype: uint8
# test_images[ i ].min(): 0
# test_images[ i ].max(): 255
# test_images[ i ].shape: (HEIGHT, WIDTH)
#
# type(test_labels):  <class 'list'>
# len(test_labels):  20000
#
# type(test_labels[ i ]): <class 'int'>
# test_labels[ i ]: 0...9
print('Reading Test 20000.cdb ...')
test_images, test_labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')

print()

# ******************************************************************************

print('type(train_images): ', type(train_images))
print('len(train_images): ', len(train_images))
print()

print('type(train_labels): ', type(train_labels))
print('len(train_labels): ', len(train_labels))
print()

fig = plt.figure(figsize=(15, 4))
for i in range(6,10):

    print('----------------------------------------')
    print()

    print('type(train_images[', i, ']):', type(train_images[i]))
    print('train_images[', i, '].dtype:', train_images[i].dtype)
    print('train_images[', i, '].min():', train_images[i].min())
    print('train_images[', i, '].max():', train_images[i].max())
    print('train_images[', i, '].shape = (HEIGHT, WIDTH):', train_images[i].shape)
    print()

    print('type(train_labels[', i, ']):', type(train_labels[i]))
    print('train_labels[', i, ']:', train_labels[i])
    print()

    fig.add_subplot(1, 4, i - 5)
    plt.title('train_labels[' + str(i) + '] = ' + str(train_labels[i]))
    plt.imshow(train_images[i], cmap='gray')

plt.savefig('train.png') 

# ******************************************************************************

print('type(test_images): ', type(test_images))
print('len(test_images): ', len(test_images))
print()

print('type(test_labels): ', type(test_labels))
print('len(test_labels): ', len(test_labels))
print()

fig = plt.figure(figsize=(15, 4))
for i in range(5,9):

    print('----------------------------------------')
    print()

    print('type(test_images[', i, ']):', type(test_images[i]))
    print('test_images[', i, '].dtype:', test_images[i].dtype)
    print('test_images[', i, '].min():', test_images[i].min())
    print('test_images[', i, '].max():', test_images[i].max())
    print('test_images[', i, '].shape = (HEIGHT, WIDTH):', test_images[i].shape)
    print()

    print('type(test_labels[', i, ']):', type(test_labels[i]))
    print('test_labels[', i, ']:', test_labels[i])
    print()

    fig.add_subplot(1, 4, i - 4)
    plt.title('test_labels[' + str(i) + '] = ' + str(test_labels[i]))
    plt.imshow(test_images[i], cmap='gray')

plt.savefig('test.png')    
plt.show()

print('################################################################################')
print()
