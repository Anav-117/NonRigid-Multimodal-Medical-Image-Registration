import SimpleITK as sitk
import numpy as np
from skimage.filters.rank import entropy, gradient
from skimage.morphology import disk
import matplotlib.pyplot as plt
import sys
import os
from math import log
from skimage import feature
from skimage.filters.thresholding import _cross_entropy


def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

def to_canny(image):
    canny_image = sitk.GetArrayFromImage(image)
    canny_image = ((canny_image/255))    
    canny_image = feature.canny(canny_image, sigma=2)
    return sitk.GetImageFromArray(canny_image.astype(float))

def to_gradient(image):
    gradient_image = sitk.GetArrayFromImage(image)
    gradient_image = ((gradient_image/255))
    gradient_image = gradient(gradient_image, disk(5))
    return sitk.GetImageFromArray(gradient_image)

def to_entropy(image):
    entropy_image = sitk.GetArrayFromImage(image)
    entropy_image = ((entropy_image))
    entropy_image = np.pad(entropy_image, ((3,3),(3,3)), 'constant', constant_values=(0, 0))
    e_image = np.zeros((len(entropy_image), len(entropy_image[0])))
    print("Starting thresholding")
    thresholds = np.arange(np.min(entropy_image) + 1.5, np.max(entropy_image) - 1.5)
    entropies = [_cross_entropy(entropy_image, t) for t in thresholds]

    optimal_threshold = thresholds[np.argmin(entropies)]
    print("Thresholding Done")
    entropy_image = (entropy_image > optimal_threshold).astype(float)

    for i in range(3, len(entropy_image)-3):
        for j in range(3, len(entropy_image[0])-3):
            window = np.zeros((7, 7))
            window_height = len(window)
            window_width = len(window[0])
            for k in range(0, len(window)):
                for l in range(0, len(window[0])):
                    window_i = i + k -(window_height/2)
                    window_j = j + l- (window_width/2)
                    window[k][l] = entropy_image[int(window_i)][int(window_j)]
            unique, counts = np.unique(window, return_counts=True)
            intensities = np.column_stack((unique, counts))
            probs = np.zeros(len(intensities))
            for a in range(0, len(intensities)):
                probs[a] = float(intensities[a][1]/9)
            sigma = 0
            for a in range(0, len(window)):
                for b in range(0, len(window[0])):
                    sigma = sigma + (probs[np.where(unique == window[a][b])])*(log(probs[np.where(unique == window[a][b])], 2))
                
            e_image[i][j] = sigma
    e_image = sitk.GetImageFromArray(e_image)    
    return e_image

def to_local_variability(image, size):
    variability_image = sitk.GetArrayFromImage(image)
    #variability_image = ((variability_image/255))
    print("Starting thresholding")
    thresholds = np.arange(np.min(variability_image) + 1.5, np.max(variability_image) - 1.5)
    entropies = [_cross_entropy(variability_image, t) for t in thresholds]

    optimal_threshold = thresholds[np.argmin(entropies)]
    print("Thresholding Done")
    variability_image = (variability_image > optimal_threshold).astype(float)
    variability_image = np.pad(variability_image, ((int((size-1)/2),int((size-1)/2)),(int((size-1)/2),int((size-1)/2))), 'constant', constant_values=(0, 0))
    v_image = np.zeros((len(variability_image), len(variability_image[0])))
    
    for i in range(int((size-1)/2), len(variability_image)-int((size-1)/2)):
        for j in range(int((size-1)/2), len(variability_image[0])-int((size-1)/2)):
            window = np.zeros((size, size))
            window_height = len(window)
            window_width = len(window[0])
            for k in range(0, len(window)):
                for l in range(0, len(window[0])):
                    window_i = i + k -(window_height/2)
                    window_j = j + l- (window_width/2)
                    window[k][l] = variability_image[int(window_i)][int(window_j)]
            unique, counts = np.unique(window, return_counts=True)
            intensities = np.column_stack((unique, counts))
            #print("\n\nINTENSITIES - ", intensities, "\n\n")
            probs = np.zeros(len(intensities))
            mu = 0
            for a in range(0, len(intensities)):
                probs[a] = float(intensities[a][1]/9)
                mu = mu + (probs[a]*intensities[a][0])
            #print("\n\nMU - ", mu, "\n\n")
            sigma = 0
            for a in range(0, len(window)):
                for b in range(0, len(window[0])):
                    sigma = sigma + (probs[np.where(unique == window[a][b])])*((mu-window[a][b])**2)
                
            v_image[i][j] = sigma
    v_image = sitk.GetImageFromArray(v_image)    
    return v_image

if len(sys.argv) < 4:
    print(
        f"Usage: {sys.argv[0]}"
        + " <fixedImageFilter> <movingImageFile> <outputTransformFile>"
    )
    sys.exit(1)

fixed_image = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
moving_image = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)

#fixed = to_entropy(fixed_image)
fixed = to_local_variability(fixed_image, 5)
#fixed = to_gradient(fixed_image)
#fixed = to_canny(fixed_image)
    
#moving = to_entropy(moving_image)
moving = to_local_variability(moving_image, 5)
#moving = to_gradient(moving_image)
#moving = to_canny(moving_image)
   
#print("FIXED - ", type(fixed))
#print("MOVING - ", type(moving))
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving, fixed)
    
# The basic Demons Registration Filter
# Note there is a whole family of Demons Registration algorithms included in
# SimpleITK
demons = sitk.DiffeomorphicDemonsRegistrationFilter()
demons.SetNumberOfIterations(1000)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(3.0)
demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
displacementField = demons.Execute(fixed, moving)

print("-------")
print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
print(f" RMS: {demons.GetRMSChange()}")

outTx = sitk.DisplacementFieldTransform(displacementField)

#sitk.WriteTransform(outTx, sys.argv[3])
sitk.WriteImage(outTx.GetDisplacementField(), sys.argv[3])
if "SITK_NOSHOW" not in os.environ:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving_image)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    simg3 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)
    simg4 = sitk.Cast(sitk.RescaleIntensity(moving_image), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    cimg2 = sitk.Compose(simg3, simg4, simg3 // 2.0 + simg4 // 2.0)
    thresholds = np.arange(np.min(sitk.GetArrayFromImage(fixed_image)) + 1.5, np.max(sitk.GetArrayFromImage(fixed_image)) - 1.5)
    entropies = [_cross_entropy(sitk.GetArrayFromImage(fixed_image), t) for t in thresholds]

    optimal_threshold = thresholds[np.argmin(entropies)]

    #fixed_image = sitk.GetImageFromArray((sitk.GetArrayFromImage(fixed_image) > optimal_threshold).astype(float))

    thresholds = np.arange(np.min(sitk.GetArrayFromImage(moving_image)) + 1.5, np.max(sitk.GetArrayFromImage(moving_image)) - 1.5)
    entropies = [_cross_entropy(sitk.GetArrayFromImage(moving_image), t) for t in thresholds]

    optimal_threshold = thresholds[np.argmin(entropies)]

    #moving_image = sitk.GetImageFromArray((sitk.GetArrayFromImage(moving_image) > optimal_threshold).astype(float))
    
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 8))
    img0 = axs[0, 0].imshow(sitk.GetArrayFromImage(fixed_image), cmap='gray')
    axs[0,0].set_title("Fixed")
    axs[0,1].imshow(sitk.GetArrayFromImage(moving_image), cmap='gray')
    axs[0,1].set_title("Moving")
    axs[0,2].imshow(sitk.GetArrayFromImage(cimg2), cmap='viridis')
    axs[0,2].set_title("Error before Registration")
    axs[0,3].imshow(sitk.GetArrayFromImage(fixed), cmap='viridis')
    axs[0,3].set_title("Fixed Gradient")
    axs[1,0].imshow(sitk.GetArrayFromImage(moving), cmap='viridis')
    axs[1,0].set_title("Moving Gradient")
    axs[1,1].imshow(sitk.GetArrayFromImage(out), cmap='viridis')
    axs[1,1].set_title("Deformed")
    axs[1,2].imshow(sitk.GetArrayFromImage(cimg), cmap='viridis')
    axs[1,2].set_title("Error after Registration")
    axs[1,3].imshow(sitk.GetArrayFromImage(outTx.GetDisplacementField())[:,:,0], cmap='viridis')
    axs[1,3].set_title("Deformation Field X")
    axs[2,0].imshow(sitk.GetArrayFromImage(outTx.GetDisplacementField())[:,:,1], cmap='viridis')
    axs[2,0].set_title("Deformation Field Y")
 
    fig.tight_layout()

    plt.show()

    #simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    #simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # Use the // floor division operator so that the pixel type is
    # the same for all three images which is the expectation for
    # the compose filter.
    #cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    #sitk.Show(outTx.GetDisplacementField(), "Deformation Field")
