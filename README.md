# NHVSampler-rs 
(WIP) A Rust UTAU resampler backends using NHVSing.  
（仍在开发）基于NHVSing的Rust UTAU重采样器后端。

This is a UTAU resampler built on [NHVSing](https://github.com/wavtechyukky/NHVSing). After noticing this vocoder, I adapted it from [hifisampler-rs](https://github.com/Slidingwall/hifisampler-rs) to evaluate its performance with UTAU.  
这是一个基于[NHVSing](https://github.com/wavtechyukky/NHVSing)的UTAU重采样器。我注意到这个声码器后，基于[hifisampler-rs](https://github.com/Slidingwall/hifisampler-rs)进行了修改，以验证其在UTAU上的表现。  

## Using 使用

> [!WARNING]
> Since this project is a quickly implemented prototype, UTAU/OpenUTAU still uses the client (`hifisampler`) for actual invocation. The server-side executable is named `nhvserver-rust`. To avoid confusion in UTAU/OpenUTAU, **it is not recommended to place both the server and the client in the `resampler` folder.**  
> 由于本项目是一个快速实现的原型，因此UTAU/OpenUTAU实际调用的客户端仍为(`hifisampler`)。作为服务器端，执行文件名为`nhvserver-rust`。为了避免在UTAU/OpenUTAU中混淆，**不建议您将服务器端与客户端一起放入`resampler`文件夹中**。  
>
> The logic for UV detection is still under development, and the current results may not be ideal.
> 判断UV的逻辑还在开发中，目前的效果可能并不理想。  

The client is as same as hifisampler. If you are using macOS or Linux, you can temporarily use the client of [StrayCat-server](https://github.com/Astel123457/straycat-server/releases/tag/release).  
客户端与原hifisampler的客户端一致。如果您使用macOS或者Linux，您可以暂时使用[StrayCat-server](https://github.com/Astel123457/straycat-server/releases/tag/release)的客户端。  

`nhvconfig.ini` is the server-side configuration file.  
`nhvconfig.ini`是服务器端的配置文件。    

You need the following ONNX models:  
您需要以下 ONNX 模型：
- [nhv_v3x](https://github.com/wavtechyukky/NHVSing/tree/main/exported_models/v3) (make sure is `nhv_v3x.onnx`, not decided to support nhv_v3 yet)

They should be located in the `./model/` folder within the same directory as the server-side, but you can also customize the model's location by modifying `nhvconfig.ini`.  
它们应位于与服务器端同目录的`./model/`文件夹内，但您也可以通过修改`nhvconfig.ini`来自定义模型的位置。  

## How to compile
 **Note**: By the nature of an UTAU resampler, it is only ideal to build this program in Windows.
 1. Install [rustup](https://rustup.rs/).
 2. Decide whether you want to build with the icon.
    - Build with icon:
        1. Install [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/).
        2. Locate `rc.exe`. It is usually in `C:\Program Files (x86)\Windows Kits\10\bin\<version number>\x64\rc.exe`
        3. Replace the location for `rc.exe` in the build script `build.rs`.
        4. Build with `cargo build -r`
    - Build without icon:
        1. Delete the build script `build.rs`.
        2. Build with `cargo build -r`
 
 I highly encourage building in the other platforms as those builds can be used in [OpenUtau.](https://github.com/stakira/OpenUtau) Build steps for Mac/Linux should be similar, just follow build without icon skipping step 1.

## Supported Flags 支持的Flags

Basically same as hifisampler-rs.  
与hifisampler-rs基本一致。

|Flags|Describe|Range|Default|
|:---:|:---:|:---:|:---:|
|**g**|Gender / formants<br/>性别 / 共振峰|-600~600|0|
|**Hb**|Breath / noise<br/>气息 / 噪波|0~500|100|
|**Hv**|Voice / harmonic<br/>发声 / 谐波|0~150|100|
|**HG**|Vocal fry / growl<br/>怒音 / 嘶吼|0~100|0|
|**P**[^1]|Note level loudness normalize<br/>音符级响度标准化|0~100|100|
|**t**|Pitch shift<br/>音高偏移|-1200~1200|0|
|**Ht**|Tension<br/>张力|-100~100|0|
|**A**|Amplitude<br/>振幅|-100~100|0|
|**G**|Force regenerate cache<br/>强制重生成缓存|bool|false|
|**He**[^2]|Loop mode<br/>循环模式|bool|false|

[^1]: Only effective when `wave_norm` is set to `true` in `nhvconfig.ini`, targeting -16 LUFS.  
      仅当`nhvconfig.ini`中，`wave_norm`为`true`时有效，以 -16 LUFS 为基准。  
[^2]: Globally enabled when `loop_mode` is set to `true` in `nhvconfig.ini`.  
      当`nhvconfig.ini`中，`loop_mode`为`true`时全局启用。  

You can download OpenUTAU resampler manifest file from [名無絃](https://bowlroll.net/file/335049).  
您可以下载[名無絃](https://bowlroll.net/file/335049)提供的OpenUTAU重采样器配置文件。  

