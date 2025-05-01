import cv2
import numpy as np

def computeGlcm(image,distance,angles=[0,45,90,135]):
    rows,cols=image.shape
    allGlcm=np.zeros((256,256,len(angles)),dtype=np.float64)

    directions={
        0: (0,distance),
        45: (-distance,distance),
        90: (-distance,0),
        135: (-distance,-distance)
    }

    for idx,angle in enumerate(angles):
        dx,dy=directions[angle]
        glcm=np.zeros((256,256),dtype=np.float64)

        for i in range(rows):
            for j in range(cols):
                ni,nj=i+dx,j+dy
                if 0<=ni<rows and 0<=nj<cols:
                    a= image[i,j]
                    b= image[ni,nj]
                    glcm[a,b]+=1
        glcm/= np.sum(glcm)
        allGlcm[:,:,idx]=glcm
    return allGlcm

def computeFeatures(allGlcm):
    features = []

    i = np.arange(256).reshape(-1, 1)
    j = np.arange(256).reshape(1, -1)
    diff = i - j
    diff2 = diff ** 2
    abs_diff = np.abs(diff)
    denom = 1 + diff2

    for idx in range(allGlcm.shape[2]):
        glcm = allGlcm[:, :, idx]
        contrast = np.sum(diff2 * glcm)
        dissimilarity = np.sum(abs_diff * glcm)
        homogeneity = np.sum(glcm / denom)
        energy = np.sum(glcm ** 2)
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
        features.append([contrast, dissimilarity, homogeneity, energy, entropy])
    return features

def extractFeaturesMultipleAngles(imagePath):
    img=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    allGlcm=computeGlcm(img,distance=1,angles=[0,45,90,135])
    features=computeFeatures(allGlcm)

    print("Features for angles 0, 45, 90, 135 degrees:")
    for angle, feature in zip([0, 45, 90, 135], features):
        print(f"Angle {angle} degrees:")
        print(f"Contrast: {feature[0]}")
        print(f"Dissimilarity: {feature[1]}")
        print(f"Homogeneity: {feature[2]}")
        print(f"Energy: {feature[3]}")
        print(f"Entropy: {feature[4]}")
        print()

if __name__ == "__main__":
    imagePath = "07DZW.png"  
    extractFeaturesMultipleAngles(imagePath)
