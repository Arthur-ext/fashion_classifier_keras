if __name__ == '__main__':
    from main import dataset, plt

    (workout_imgs, workout_identifiers), (test_img, test_identifier) = dataset.load_data()
    
    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(ld.workout_imgs[i])
    #     plt.title(ld.ids[ld.workout_identifiers[i]])
    #     plt.colorbar()
    print(workout_imgs[59001])
    plt.imshow(workout_imgs[59001])
    plt.colorbar()

    plt.show()