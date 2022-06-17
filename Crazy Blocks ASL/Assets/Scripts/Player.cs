using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Rendering;

public class Player : MonoBehaviour
{
    [Header("Scene Objects")]
    [SerializeField] Vector2 jumpVelocity = new Vector2(0f, 1f);

    [Header("Data Processing")]
    [Tooltip("The target minimum model input dimensions")]
    public int targetDim = 216;
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("The material with the fragment shader for GPU processing")]
    public Material processingMaterial;

    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The name for the custom softmax output layer")]
    public string softmaxLayer = "softmaxLayer";
    [Tooltip("The name for the custom softmax output layer")]
    public string argmaxLayer = "argmaxLayer";
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Header("Output Processing")]
    [Tooltip("Asynchronously download model output from the GPU to the CPU.")]
    public bool useAsyncGPUReadback = true;
    [Tooltip("A json file containing the class labels")]
    public TextAsset classLabels;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;

    [Header("Webcam")]
    [Tooltip("Use a webcam as input")]
    public bool useWebcam = false;
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);
    [Tooltip("The requested webcam framerate")]
    [Range(0, 60)]
    public int webcamFPS = 60;

    [Header("GUI")]
    [Tooltip("Display predicted class")]
    public bool displayPredictedClass = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0, 99)]
    public int fontScale = 50;


    // List of available webcam devices
    private WebCamDevice[] webcamDevices;
    // Live video input from a webcam
    private WebCamTexture webcamTexture;
    // The name of the current webcam  device
    private string currentWebcam;

    // The current screen object dimensions
    private Vector2Int screenDims;
    // The model input texture
    private RenderTexture inputTexture;

    // The main interface to execute models
    private IWorker engine;
    // Stores the input data for the model
    private Tensor input;

    // Stores the raw model output on the GPU when using useAsyncGPUReadback
    private RenderTexture outputTextureGPU;
    // Stores the raw model output on the CPU when using useAsyncGPUReadback
    private Texture2D outputTextureCPU;

    // A class for reading in class labels from a JSON file
    class ClassLabels { public string[] classes; }
    // The ordered list of class names
    private string[] classes;
    // Stores the predicted class index
    private int classIndex;

    private Vector3 initPosition;


    /// <summary>
    /// Initialize the selected webcam device
    /// </summary>
    /// <param name="deviceName">The name of the selected webcam device</param>
    private void InitializeWebcam(string deviceName)
    {
        // Stop any webcams already playing
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();
        Debug.Log($"Selected Webcam: {deviceName}");
        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(deviceName, webcamDims.x, webcamDims.y, webcamFPS);

        // Start the webcam
        webcamTexture.Play();
        // Check if webcam is playing
        //useWebcam = webcamTexture.isPlaying;
        //// Update toggle value
        //useWebcamToggle.SetIsOnWithoutNotify(useWebcam);

        Debug.Log(webcamTexture.isPlaying ? "Webcam is playing" : "Webcam not playing");
    }


    /// <summary>
    /// Initialize an interface to execute the specified model using the specified backend
    /// </summary>
    /// <param name="model">The target model representation</param>
    /// <param name="workerType">The target compute backend</param>
    /// <param name="useNCHW">EXPERIMENTAL: The channel order for the compute backend</param>
    /// <returns></returns>
    private IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW = true)
    {
        // Validate the selected worker type
        workerType = WorkerFactory.ValidateType(workerType);

        // Set the channel order of the compute backend to channel-first
        if (useNCHW) ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;

        // Create a worker to execute the model using the selected backend
        return WorkerFactory.CreateWorker(workerType, model);
    }


    // Start is called before the first frame update
    void Start()
    {
        initPosition = this.transform.position;

        // Initialize list of available webcam devices
        webcamDevices = WebCamTexture.devices;
        foreach (WebCamDevice device in webcamDevices) Debug.Log(device.name);
        currentWebcam = webcamDevices[0].name;
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        // Initialize webcam
        if (useWebcam) InitializeWebcam(currentWebcam);


        // Get an object oriented representation of the model
        Model m_RunTimeModel = ModelLoader.Load(modelAsset);
        // Get the name of the target output layer
        string outputLayer = m_RunTimeModel.outputs[outputLayerIndex];

        // Create a model builder to modify the m_RunTimeModel
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

        // Add a new Softmax layer
        modelBuilder.Softmax(softmaxLayer, outputLayer);
        // Add a new Argmax layer
        modelBuilder.Reduce(Layer.Type.ArgMax, argmaxLayer, softmaxLayer);
        // Initialize the interface for executing the model
        engine = InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Initialize the GPU output texture
        outputTextureGPU = RenderTexture.GetTemporary(1, 1, 24, RenderTextureFormat.ARGBHalf);
        // Initialize the CPU output texture
        outputTextureCPU = new Texture2D(1, 1, TextureFormat.RGBAHalf, false);

        // Initialize list of class labels from JSON file
        classes = JsonUtility.FromJson<ClassLabels>(classLabels.text).classes;
    }

    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image">The target image RenderTexture</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = computeShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    private Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        // Clamp the minimum dimension value to 64px
        targetDim = Mathf.Max(targetDim, 64);

        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions using the target minimum dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }


    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Make sure the Texture2D is not null
        if (outputTextureCPU)
        {
            // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            // Apply changes to Textur2D
            outputTextureCPU.Apply();
        }
    }


    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    private int ProcessOutput(IWorker engine)
    {
        int classIndex = -1;

        // Get raw model output
        Tensor output = engine.PeekOutput(argmaxLayer);

        if (useAsyncGPUReadback)
        {
            // Copy model output to a RenderTexture
            output.ToRenderTexture(outputTextureGPU);
            // Asynchronously download model output from the GPU to the CPU
            AsyncGPUReadback.Request(outputTextureGPU, 0, TextureFormat.RGBAHalf, OnCompleteReadback);
            // Get the predicted class index
            classIndex = (int)outputTextureCPU.GetPixel(0, 0).r;

            // Check if index is valid
            if (classIndex < 0 || classIndex >= classes.Length) Debug.Log("Output texture not initialized");
        }
        else
        {
            // Get the predicted class index
            classIndex = (int)output[0];
        }

        if (printDebugMessages) Debug.Log($"Class Index: {classIndex}");

        // Dispose Tensor and associated memories.
        output.Dispose();

        return classIndex;
    }


    // Update is called once per frame
    void Update()
    {

        //useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        if (useWebcam)
        {
            // Initialize webcam if it is not already playing
            if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

            // Skip the rest of the method if the webcam is not initialized
            //if (webcamTexture.width <= 16) return;
        }

        // Scale the source image resolution
        Vector2Int inputDims = CalculateInputDims(webcamDims, targetDim);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

        // Initialize the input texture with the calculated input dimensions
        inputTexture = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputTexture.width}x{inputTexture.height}");

        // Copy the source texture into model input texture
        Graphics.Blit(webcamTexture, inputTexture);

        // Disable asynchronous GPU readback when not using a Compute Shader backend
        useAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel") ? useAsyncGPUReadback : false;

        if (SystemInfo.supportsComputeShaders)
        {
            // Normalize the input pixel data
            ProcessImageGPU(inputTexture, processingShader, "NormalizeImageNet");

            // Initialize a Tensor using the inputTexture
            input = new Tensor(inputTexture, channels: 3);
        }
        else
        {
            // Define a temporary HDR RenderTexture
            RenderTexture result = RenderTexture.GetTemporary(inputTexture.width,
                inputTexture.height, 24, RenderTextureFormat.ARGBHalf);
            RenderTexture.active = result;

            // Apply preprocessing steps
            Graphics.Blit(inputTexture, result, processingMaterial);

            // Initialize a Tensor using the inputTexture
            input = new Tensor(result, channels: 3);
            RenderTexture.ReleaseTemporary(result);
        }

        // Execute the model with the input Tensor
        engine.Execute(input);
        // Dispose Tensor and associated memories.
        input.Dispose();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTexture);
        // Get the predicted class index
        classIndex = ProcessOutput(engine);
        // Check if index is valid
        
    }

    private void FixedUpdate()
    {
        bool validIndex = classIndex >= 0 && classIndex < classes.Length;
        if (printDebugMessages) Debug.Log(validIndex ? $"Predicted Class: {classes[classIndex]}" : "Invalid index");

        // Unload assets when running in a web browser
        if (Application.platform == RuntimePlatform.WebGLPlayer) Resources.UnloadUnusedAssets();


        if (Input.GetButtonDown("Fire1") || (validIndex && classes[classIndex] == "B"))
        {
            GetComponent<Rigidbody2D>().velocity = jumpVelocity;
            //GetComponent<Rigidbody2D>().angularVelocity = jumpVelocity.y;

        }
        if (Input.GetKey(KeyCode.Escape) || (validIndex && classes[classIndex] == "Stop")) Application.Quit();
    }

    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        // Define styling information for GUI elements
        GUIStyle style = new GUIStyle
        {
            fontSize = (int)(Screen.width * (1f / (100f - fontScale)))
        };
        style.normal.textColor = textColor;

        // Define screen spaces for GUI elements
        Rect slot1 = new Rect(10, 10, 500, 500);

        // Verify predicted class index is valid
        bool validIndex = classIndex >= 0 && classIndex < classes.Length;
        string content = $"Predicted Class: {(validIndex ? classes[classIndex] : "Invalid index")}";
        if (displayPredictedClass) GUI.Label(slot1, new GUIContent(content), style);
    }


    public void Cleanup()
    {
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();
        Destroy(webcamTexture);

        // Release the resources allocated for the outputTextureGPU
        RenderTexture.ReleaseTemporary(outputTextureGPU);

        // Release the resources allocated for the inference engine
        engine.Dispose();

    }


    public void ResetPosition()
    {
        this.transform.position = initPosition;
    }

    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        Cleanup();
    }

    //// OnDisable is called when the MonoBehavior becomes disabled
    //private void OnDestroy()
    //{
    //    // Release the resources allocated for the outputTextureGPU
    //    RenderTexture.ReleaseTemporary(outputTextureGPU);

    //    // Release the resources allocated for the inference engine
    //    engine.Dispose();

    //    if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();
    //}
}
