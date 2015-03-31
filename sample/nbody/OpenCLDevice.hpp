
#include <iostream>
#include <fstream>
#include <sstream>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>

typedef unsigned long long int uint64;
typedef unsigned int           uint32;

class OpenCLDevice {
public:
  OpenCLDevice(int);
  ~OpenCLDevice();
  void SetupContext(int);
  void SetKernelOptions(std::string);
  void BuildOpenCLKernels(const char *, const char *);
  cl::Kernel GetKernel(const char *);
  cl::Context ctx;
  cl::CommandQueue q;
protected:
  uint64 n_cpus;
  uint64 n_gpus;
  cl::Platform pf;
  cl::Device dev;
  cl::Program prog;
  std::vector<cl::Device> v_dev;
  std::string kernel_options;
};

OpenCLDevice::OpenCLDevice(int ip)
{
  try {
    std::vector<cl::Platform> pfs;
    cl::Platform::get(&pfs);
    if (pfs.size() <= (uint64)ip) throw cl::Error(-1, "FATAL: the specifed platform does not exist");
    pf = pfs[ip];
    std::cerr << pf.getInfo<CL_PLATFORM_NAME>().c_str() << " "
	      << pf.getInfo<CL_PLATFORM_VERSION>().c_str() << "::";    
  }  catch( cl::Error e ) {
    std::cerr << e.what() << ":" << e.err() << "\n";
    exit(-1);
  }
}

void OpenCLDevice::SetKernelOptions(std::string options)
{
  kernel_options = options;
}

void OpenCLDevice::SetupContext(int id)
{
  try {
    std::vector<cl::Device> devs;
    pf.getDevices(CL_DEVICE_TYPE_ALL, &devs);
    if (devs.size() <= (uint64)id) throw cl::Error(-1, "FATAL: the specifed device does not exist");
    dev = devs[id];
    std::cerr << dev.getInfo<CL_DEVICE_NAME>().c_str() << "\n";
      
    v_dev.push_back(dev);
    ctx = cl::Context(v_dev);
    q   = cl::CommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
  }  catch( cl::Error e ) {
    std::cerr << e.what() << ":" << e.err() << "\n";
    exit(-1);
  }
}

void OpenCLDevice::BuildOpenCLKernels(const char *kernelfile, const char *more_options = NULL) 
{
  try {
    cl::Program::Sources src(1, std::make_pair(kernelfile, strlen(kernelfile)));
    prog = cl::Program(ctx, src);

    std::stringstream options;

    if (more_options != NULL) 
      options << more_options;

    options << kernel_options;

    std::cerr << "Build options :: " << options.str() << "\n";
    prog.build(v_dev, options.str().c_str());
  }  catch( cl::Error e ) {
    std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
    std::cerr << e.what() << ":" << e.err() << "\n";
    std::cerr << kernelfile << "\n";
    std::cerr << log << "\n";
    exit(-1);
  }
}

cl::Kernel OpenCLDevice::GetKernel(const char *kernel_main)
{
  return cl::Kernel(prog, kernel_main);
}
