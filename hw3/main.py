from PIL import Image;
import numpy as np

def getRGB(image):
    im = Image.open(image);
    pix = im.load();
    width = im.size[0];
    height = im.size[1];
    rgb = [];
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y];
            rgb.append([r/256.0, g/256.0, b/256.0]);


    return(rgb, width, height)

def kMeans(imageRGB, nCluster):
    randoms = np.random.choice(len(imageRGB), nCluster, replace = False);
    idx = np.zeros(len(imageRGB));
    dist = np.zeros(nCluster);
    means = [];
    for random in randoms:
        means.append(imageRGB[random]);
    
    iters = 5

    for i in range(iters):
        for j in range(len(imageRGB)):
            for k in range(nCluster):
                imageM = np.asarray(imageRGB[j]);
                meansM = np.asarray(means[k])
                dist[k] = np.linalg.norm(imageM - meansM);

            idx[j] = np.argmin(dist)


        for i in range(nCluster):
            cluster = np.empty((0, 3));
            for j in range(len(imageRGB)):
                
                if (idx[j] == i):
                    print(np.array([imageRGB[j]]))
                    cluster = np.append(cluster, np.array([imageRGB[j]]), axis = 0);
            means[i] = np.array([np.mean(cluster[:,0]),np.mean(cluster[:,1]),np.mean(cluster[:,2])]);

    return means, idx;



images = ["4.jpg"];
clusters = [5]

for image in images:
    for nCluster in clusters:
        imageRGB, width, height = getRGB(image);

        means, idx = kMeans(imageRGB, nCluster);
        newMeans = np.asarray(means);
        newImage = newMeans[idx.astype('uint8')];
        newImage = (256 * newImage).astype('uint8');

        newImage = np.reshape(newImage, (width, height, 3));
        newImage = Image.fromarray(newImage);
        newImageName = image[:-4] + "_" + str(nCluster) + ".jpg";
        newImage.save(newImageName);