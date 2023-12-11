# Accelerating CPU Inference with OpenVINO on Intel Developer Cloud

**Julian Hernandez**  
*UC Berkeley*  
*Department of Civil and Environmental Engineering*  
*Berkeley, California*

## Introduction

Traffic monitoring has been an increasingly important aspect in transportation research with the development of electronic sensors. In California, the first traffic sensors were installed in the 1970s, and since then Caltrans has installed over 40,000 more traffic sensors across the 50,000-mile highway network [1]. They’ve also expanded the Transportation Monitoring System (TMS) to nine types of sensors including closed circuit television (CCTV), highway advisory radios, and roadway weather information systems. Together these gather a wide array of information about roadway conditions, and allow that information to get to active drivers on the roadway. Some of the data from traffic sensors is able to be automatically processed using the Performance Management System (PeMS) which was developed at UC Berkeley at the Partners for Transit and Highways Research Center [2]. This helps collect aggregated traffic information about speed, flows and density, using under the road sensors. However, the CCTV video streams must still be manually reviewed. With recent advances in artificial intelligence (AI) the ability to track cars and detect incidents or congestion, these cameras could be used to increase data fidelity across the network. Computer vision models are being tested by various Departments of Transportation, but none had implemented statewide coverage, and handled orders of magnitude fewer cameras than what Caltrans currently manages [3]. This project seeks to measure the computational complexity of running both a state-of-the-art object detection model and incident detector, on a statewide level.

## Dataset

Many state DOTs such as California and New York have large CCTV camera arrays that make their footage publicly available using live streams. This provides a source of real-time data that can be used as a benchmark for computational complexity analysis. For example, the California Department of Transportation has installed hundreds of cameras across the state that are connected to the internet using fiber optic cables that run along the highway right of way [4]. While these cameras are natively 1080p, the streams are usually downscaled to 240p 360p, which makes use of computer vision technology difficult to implement, and less accurate than other solutions. Thus, for the purposes of this study, the NY511 CCTV network was selected as a baseline [5]. This dataset contains over 1,600 cameras that are across the state of New York, and have higher quality video streams, ranging from 320x320 to 2560x2560 pixels (Figure 1).

![Figure 1: Count of cameras by resolution](figures/figure 1.png)

A previous study by the Center for Advanced Infrastructure and Transportation at Rutgers University also used this dataset for performing AI car detection in 2019 [6]. However, in the past 4 years AI has made great strides in improving accuracy and speed, while computational hardware, especially cloud-based hardware, has grown exponentially faster with the development of specialized matrix calculators. Although their study looked into using cloud computing through AWS for processing the video streams, they only looked at a few videos, and thus didn’t provide a good estimate for overall computational complexity of running inference on the system. This study will fill in that gap by looking at processing a one-minute slice of video from every available stream, then using this as a baseline, to determine how many processors would be needed to run the system in real-time.

Because of the large size of video content, especially for live streaming, these files are compressed which means that even videos with the same dimensions and time can have highly different data rates per minute. The video live streams are hosted using the m3u8 protocol, which splits the video up into downloadable chunks that can be converted to MP4. These are downloaded then all the chunks are merged together to be one minute long, then the file sizes are summed to find the total data rate per minute.

![Figure 2: Stream data rate in megabytes per minute](/figures/figure 2.png)

In figure 2 we can see that most of the video streams are in the range of one to two Megabytes per minute, resulting in a total of 9.805 Gigabytes per minute for the whole system. To save all the footage from these cameras for one day would require 13.788 Terabytes of storage. This highlights just how large CCTV datasets can be, which is why only one minute of footage will be analyzed to provide a total system estimate, as the costs to process Terabytes of data are out of the scope of this project.

![Figure 3: Count of cameras by frames per second (FPS)](figures/figure 3.png)

Perhaps the most important aspect of this dataset is how many frames are produced each minute. This can be calculated by extracting the FPS of each camera for the dataset and multiplying by 60 to convert seconds to minutes, resulting in 865,920 frames per minute for the whole system. Using this same calculation but converting to a per day basis we can find that the system outputs 1,246,924,800 frames per day. This analysis highlights the importance of computational efficiency for live video streams, as only one DOT will need to process billions of images per day.

## Computer Vision Model

In order to efficiently process this dataset a very fast computer vision model will need to be used. For the purposes of this study You Only Look Once version 8 (YOLOv8) was selected as it is one of the fastest state of the art models for object detection for vehicles [7]. The YOLO family of detectors first came onto the scene in 2015, pioneered by Joseph Redmon, and quickly gained popularity due to its improved speed without sacrificing accuracy. Each year new versions have been released that build on his initial model, and today Ultralytics, an AI company based in Spain, has taken over development and maintenance of versions 5 and 8 of the model. They provide an easy-to-use Python package that combines image preprocessing, inference, and postprocessing into one `predict()` function. YOLOv8 comes in 5 sizes: nano, small, medium, large, and extra-large. These provide the selection between accuracy and processing time, for this study we chose the nano-sized model to measure peak efficiency when using a highly optimized model.

## CPU Inference

Although traditionally computer vision models have been run on Graphics Processing Units (GPUs) which provide better parallelization, in recent years Central Processing Units (CPUs) have become more and more suited to AI tasks, especially for inference. The gold standard in AI is to use Nvidia A100s in the cloud for training and inference, but these are becoming increasingly hard to come by and prohibitively expensive. [8] Because of these limitations, researchers are turning to CPUs for their inference tasks, where latency might be more important than throughput, and implementing AI in your core business logic is much easier.

![Figure 4: Ultralytics Benchmark [9]](<Image URL>)

Ultralytics has already provided benchmarks for their models on an Nvidia A100 and a second-generation Intel Xeon Scalable processor. Looking at their results we can see that CPU inference is roughly 1/80th as fast as inference on the GPU, however this doesn’t tell the full story. CPU inference is only on a per-core basis, and with 56 cores the latest Xeon processors can process many streams of data at once. Additionally, in the past CPUs weren’t optimized for matrix operations that are used in neural networks, but Intel’s 4th Generation Xeon CPUs now contain Advance Matrix Accelerators (AMX) that mimics the way GPUs process AI workloads. In order to get access to these expensive processors inference was run in the Intel Developer Cloud which provides free testing access to these CPUs for testing AI workloads on Intel hardware.

Three Intel CPUs will be used in this study and benchmarked against the A100 speeds seen in figure 4. These will include a laptop mobile processor with iGPU (For 13th Gen Intel® Core™ i5-1340P), a desktop processor without a GPU (For 13th Gen Intel® Core™ i7-13700F) and an Intel Xeon processor in the cloud (For Intel ® Xeon ® Platinum 8480+).

## Hardware Acceleration with OpenVINO

OpenVINO will be used to unlock the full potential of these CPUs since frameworks such as PyTorch have been optimized for GPUs and don’t include the most optimized libraries for CPU inference. OpenVINO comes with two core workflows, a model optimizer and inference engine. First, a model is trained using any popular framework such as PyTorch or Caffe, this study uses a pre-trained model from Ultralytics using PyTorch. Then the model optimizer converts this into a format that is computationally efficient for loading into your CPUs cache. This optimized model can be run on the inference engine that will utilize the full capabilities of the CPU including any iGPUs or accelerators if available.

![Figure 5: OpenVINO workflow [10]](<Image URL>)

## Results

All frames from the cameras were processed using each processor, once using the base PyTorch model and once using an optimized OpenVINO version of the same model. These results show single-core processing time but each CPU has multiple cores that allow for parallel processing that mimics the way a GPU would function. The figures below show histograms of inference time for each CPU between PyTorch and OpenVINO, where lower scores are better.

**Figures 5: Inference Speed on Intel Mobile CPU using PyTorch**  
**Figure 6: Inference Speed on Intel Mobile CPU using OpenVINO**  
**Figures 7: Inference Speed on Intel Desktop CPU using PyTorch**  
**Figure 8: Inference Speed on Intel Desktop CPU using OpenVINO**  
**Figures 9: Inference Speed on Intel Xeon CPU using PyTorch**  
**Figure 10: Inference Speed on Intel Xeon CPU using OpenVINO**

From figures 5 and 6, we can see a nearly 10x performance boost on the laptop CPU when optimizing using OpenVINO. This huge increase is likely due to using the iGPU to help accelerate the workload. While the mean inference time decreases from 342ms to 33ms we can also see that the distribution of the data converges closer to the mean, providing better reliability when estimating scaling costs. The desktop platform also sees a decrease in processing time from 107ms to 66ms, this decrease is much less than with the mobile CPU and after optimizing with OpenVINO we see that the lower end mobile platform outperforms the desktop processor. This is because the desktop processor doesn’t have any AI accelerators or iGPU to utilize, so although openVINO can help optimize the model, the inference engine has less architectural support to take advantage of. The best results are seen with the Intel Xeon Scalable processors that contain AMX hardware on the chip. Here we see a decrease from 23ms to 6ms when using OpenVINO, which is coming close to the 1ms provided by the Nvidia A100.

Now to calculate how many processors would be needed to run object detection on the full video dataset we can run this calculation for each processor:

