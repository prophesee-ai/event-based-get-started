# Getting started with Metavision
Easy examples for event processing based on Prophesee **Metavision SDK 4.6.X** (4.6.0 and further minor updates).

This repository provides a few C++ and Python examples demonstrating ways to retrieve events from a recording or a live 
camera and process them to do visualization, processing, or AI. It is meant to facilitate integration of event-based 
data in your programmation pipelines, to develop POCs, or to help you discover the potential of event-based technology.

For more information about event processing or tips to build your event-based system, have a look at this  
[application note](https://support.prophesee.ai/portal/en/kb/articles/how-to-build-event-based-application) which 
provides both high-level tips and very practical information including development basis and common practices.

## Examples

### Time Surface viewer (Available with **OpenEB**)
This first sample is a simple timesurface visualization of events. A timesurface is a matrix storing at each pixel the
timestamp of the last event triggered at this location. It can be further used for processing, such as finding edges in
the image, computing the optical flow etc. Both C++ and Python samples are provided.
![Time surface example.](./images/time_surface.png "This is a time surface.")

### Dummy Radar (Available with **Metavision SDK**)
This C++ sample provides some very easy visualization for localizing the most prominent movement in the scene within
some event rate window. Typically, it allows to localize a person in the camera field of view. The object "distance"
is estimated relatively to the provided event rates: the closer the person, the more events it will generate.
![Radar example.](./images/radar_plot.png "Radar display of the camera observation.")

### Image sharpener from ML Optical Flow (Available with **Metavision SDK**)
This last Python sample computes the optical flow from input events and uses this flow to apply a sharpening function
to the live-built image representation of events.
![Sharpening example.](./images/image_sharpening.png "Image sharpening from Optical Flow.")

### Data
Some example data is available [here](https://kdrive.infomaniak.com/app/share/975517/9bb88895-ab07-4bfc-8b31-f71de075175c).
It features two recordings, as well as an example of camera configuration file which you can provide to some samples.

## Usage
This repository expects a compiled version of Metavision SDK, either a custom local build or an installed version, which 
is available in the [Metavision SDK Documentation](https://docs.prophesee.ai/stable/index.html).

For the C++ samples, you will need to create your build folder and compile the sample from there. Remember to provide 
Metavision paths if you are working with a local Metavision build.

```
cmake .. -DMetavisionSDK_DIR=<METAVISION_BUILD_DIR>/generated/share/cmake/MetavisionSDKCMakePackagesFilesDir/ 
-DMetavisionHAL_DIR=<METAVISION_BUILD_DIR>/generated/share/cmake/MetavisionHALCMakePackagesFilesDir/ 
-DCMAKE_BUILD_TYPE=Release

make 
```

If you are working with a local Metavision build, you will need to source the setup file before running one of the apps.
```
source <METAVISION_BUILD_FOLDER>/utils/script/setup_env.sh
```

Then, you can run the demos:
```
./bin/metavision_time_surface # On Ubuntu
./bin/dummy_radar             # On Ubuntu
./bin/metavision_time_surface.exe # On Windows
./bin/dummy_radar.exe             # On Windows
```

For Python ones:
```
python3 metavision_time_surface.py # Ubuntu
python metavision_time_surface.py  # Windows
```

A configuration file is provided in the **data** folder as an example. In particular, it allows to enable the hardware 
STC to filter noise and trails of events. It can be provided to some samples as a command line argument. 
```
"event_trail_filter_state": {
 "enabled": true,
 "filtering_type": "STC",
 "threshold": 1000
}
```
