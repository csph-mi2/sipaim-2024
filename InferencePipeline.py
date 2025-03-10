import argparse
import pdb
import onnxruntime
import numpy as np
from os import path, makedirs
import torch
import SimpleITK as sitk
import vtk

suture_fusion_options = ['Normative','Metopic','Sagittal','Unicoronal_right','Unicoronal_left']
IMAGE_MODEL_DIR = r'.\models\master.onnx'
RANDOM_MODEL_DIR = r'.\models\master-random.onnx'

def GenerateRandomNoise():
    random_noise = np.random.randn(1, 64)
    return random_noise

def EncodeSutureFusion(suture_fusion):
    idx = suture_fusion_options.index(suture_fusion)
    res = np.zeros((4))
    if idx != 0:
        res[idx-1] = 1
    return res


def ParseInputImage(input_image):
    if input_image is None:
        return GenerateRandomNoise()
    elif input_image.endswith('.pt'):
        return torch.load(input_image)
    elif input_image.endswith('.mha'):
        return sitk.ReadImage(input_image)
    else:
        raise ValueError('Input image must be an mha or None, which will use random noise.')

def EncodeAge(age):
    #conver the age to years, normalized to 10
    # 1 == 10 years (max of our model)
    new_age = age/(365*10)
    if new_age > 1:
        new_age = 1
        print('Age is larger than 10 years. Setting inference to 10 years.')
    return new_age

def EncodeSex(sex):
    return 1 if sex == 'M' else 0

def WritePolyData(data, filename):
    if filename.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    # Saving landmarks
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Update()
    return

def CreateMeshFromBullsEyeImageWithCoords(euclideanCoordinateSphericalMapImage, intensityImageDict=None, subsamplingFactor=1, verbose=False):
    """
    Recsontructs a surface model using the Euclidean coordinates represented as a spherical map image 

    Parameters
    ----------
    euclideanCoordinateSphericalMapImage: sitkImage
        Spherical map image with the Euclidean coordinates of the surface model
    referenceImage: sitkImage
        A reference image with with pixels set to 0 in the background
    intensityImageDict: dictionary {arrayName: image}
        A dictionary

    Returns
    -------
    sitkImage:
        The reconstructed image    
    """
    background_value = euclideanCoordinateSphericalMapImage.GetPixel(0,0)[0]

    euclideanCoordinateSphericalMapImage.SetSpacing((1,1))
    euclideanCoordinateSphericalMapImage.SetOrigin((0,0))

    filter = vtk.vtkPlaneSource()
    filter.SetOrigin((0, 0, 0))
    filter.SetPoint1((1 * euclideanCoordinateSphericalMapImage.GetSize()[0], 0, 0))
    filter.SetPoint2((0, 1 * euclideanCoordinateSphericalMapImage.GetSize()[1], 0))
    filter.SetXResolution(int((euclideanCoordinateSphericalMapImage.GetSize()[0]) / subsamplingFactor))
    filter.SetYResolution(int((euclideanCoordinateSphericalMapImage.GetSize()[1]) / subsamplingFactor))
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    insideArray = vtk.vtkIntArray()
    insideArray.SetName('Inside')
    insideArray.SetNumberOfComponents(1)
    insideArray.SetNumberOfTuples(mesh.GetNumberOfPoints())



    intensityImageArrays = []
    intensityArrays = []

    if intensityImageDict is not None:

        for key, val in intensityImageDict.items():

            intensityImageArrays += [np.array(sitk.GetArrayFromImage(val))]
            intensityArrays += [vtk.vtkFloatArray()]
            intensityArrays[-1].SetName(key)
            intensityArrays[-1].SetNumberOfComponents(3)
            intensityArrays[-1].SetNumberOfTuples(mesh.GetNumberOfPoints())
            mesh.GetPointData().AddArray(intensityArrays[-1])

    # Figuring out what is inside or outside
    points_placed = 0
    for p in range(mesh.GetNumberOfPoints()):
            
        if verbose:
            print('{} / {}.'.format(p, mesh.GetNumberOfPoints()), end='\r')

        coords = mesh.GetPoint(p)
        try:
            imageIndex = euclideanCoordinateSphericalMapImage.TransformPhysicalPointToIndex((coords[0], coords[1]))
            
            mesh.GetPoints().SetPoint(p, euclideanCoordinateSphericalMapImage.GetPixel(imageIndex))
            #don't include this point if its 0,0
            # if (referenceImageArray[imageIndex[1], imageIndex[0]] == 1):
            if not (np.array([x==background_value for x in euclideanCoordinateSphericalMapImage.GetPixel(imageIndex)]).all()):
                points_placed+=1
                insideArray.SetTuple1(p, 1)
                if intensityImageDict is not None:
                    for arrayId in range(len(intensityArrays)):
                        intensityArrays[arrayId].SetTuple3(p, *intensityImageArrays[arrayId][imageIndex[1], imageIndex[0]])
            else:
                insideArray.SetTuple1(p, 0)
        except:
            #this just means the coordinate is at the end of the image and it is undefined
            insideArray.SetTuple1(p, 0)

    mesh.GetPointData().AddArray(insideArray)

    filter = vtk.vtkThreshold()
    filter.SetInputData(mesh)
    filter.SetLowerThreshold(0.1)
    filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'Inside')
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(mesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SplittingOff()
    filter.Update()
    mesh = filter.GetOutput()

    mesh.GetPointData().RemoveArray("Inside")

    return mesh

def evaluate_model(input_image = None, age = 0, sex = 1, suture_fusion = [0,0,0,0],outpath = None):
    if outpath is None:
        outpath = path.join('.','Evaluation','mesh.vtp')
    if not path.exists(path.dirname(outpath)):
        makedirs(path.dirname(outpath))

    # construct y
    y = np.zeros((1,6))
    y[:,0] = age
    y[:,1] = sex
    y[:,2:] = suture_fusion
    if type(input_image) == torch.Tensor:
        input_image = input_image.numpy()

    if input_image.ndim == 2:
        #this is random noise
        MODEL_DIR = RANDOM_MODEL_DIR
        input_image = GenerateRandomNoise()
    elif input_image.ndim == 3:
        #this is a real image
        MODEL_DIR = IMAGE_MODEL_DIR

    #TODO:
    '''
    1. Set up a class that contains all of our models
    2. Set up a function that takes in raw data!
    3. Set up a function that generates the output from model inference
    '''

    '''
    Try with the encoder/predictor
    '''

    ort_session = onnxruntime.InferenceSession(MODEL_DIR, providers=['CPUExecutionProvider'])
    #need float 32 here because that is the default for pytorch
    onnxruntime_input = {k.name: v.astype(np.float32) for k, v in zip(ort_session.get_inputs(), (input_image,y))}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    '''
    convert to a simpleitk image
    '''
    image_res = sitk.GetImageFromArray(onnxruntime_outputs[0].astype(np.float32), isVector=True)
    origin = (0.0, 0.0)
    spacing = (1.0, 1.0)
    direction = (1.0, 0.0, 0.0, 1.0)
    image_res.SetOrigin(origin)
    image_res.SetSpacing(spacing)
    image_res.SetDirection(direction)
    
    target_mesh = CreateMeshFromBullsEyeImageWithCoords(image_res)
    WritePolyData(target_mesh, outpath)

def ConstructArguments():
    parser = argparse.ArgumentParser(description='Process 3D photogram')
    ## Required arguments
    parser.add_argument('--age', required = True, type = float, metavar = 'age',
        help='Age of the patient in days.')

    parser.add_argument('--sex', required = True, type = str, metavar = 'sex', choices = ['M','F'], 
        help='Sex of the patient (F is female, M is male).')
    
    parser.add_argument('--suture_fusion', required = True, type = str, metavar = 'suture_fusion', choices = suture_fusion_options, help = 'Condition of pathology.')

    parser.add_argument('--output_path', required = False, type = str, metavar = 'output_path',
        help = 'Path to the output file.')

    parser.add_argument('--input_image', required = False, type = str, metavar = 'input_image', default = None,
        help = 'Path to the input image.')
    
    return parser

def ParseArguments(argv=None):
    parser = ConstructArguments()
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    #now parse the input image, sex, age, suture fusion
    args.input_image = ParseInputImage(args.input_image)
    args.sex = EncodeSex(args.sex)
    args.age = EncodeAge(args.age)
    args.suture_fusion = EncodeSutureFusion(args.suture_fusion)
    return args

if __name__ == "__main__":
    logdir = r'C:\Users\elkhillc\OneDrive - The University of Colorado Denver\MIP\PredictiveModel\SIPAIM-2024'
    # argv = ['--age', '100', '--sex', 'M', '--suture_fusion', "Normative", '--output_path', path.join(logdir,'Evaluation','mesh.vtp')]
    args = ParseArguments()
    #convert from image to the dataset we want
    # if args.input_image is not None:
    #     ex_data = torch.load(args.input_image)
    #     image = ex_data[0].numpy()
    for age in range(1,100):
        args.output_path = path.join(logdir,'Evaluation','mesh_{}.vtp'.format(age))
        evaluate_model(input_image = args.input_image, age = args.age, sex = args.sex, suture_fusion = args.suture_fusion, outpath = args.output_path)