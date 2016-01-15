namespace NeuThink
module Native =
 open System
 open System.Runtime.InteropServices
 open Microsoft.FSharp.NativeInterop 

 let buffer = Array.init 500 (fun i -> 0.0) 

 [<System.Runtime.InteropServices.DllImport(@"native_func.dll",EntryPoint="vectmat")>]
 extern void vectmat(double *proc_inputs,double *weights,double *outputs,int psize,int wsize,int osize)  

 [<System.Runtime.InteropServices.DllImport(@"libacml_mp_dll.dll",EntryPoint="dgemv")>]
 extern void dgemv(char transa, int m, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy);


 let inline (~~) (data : GCHandle) = data.AddrOfPinnedObject()
 let inline (~~~) (data : GCHandle) = NativePtr.ofNativeInt (data.AddrOfPinnedObject())
 let inline (!~) (ptr  : GCHandle) = ptr.Free()
 let pin (data : double array)       = GCHandle.Alloc(data,GCHandleType.Pinned)

 let vect_mat (weights:float array) (proc_inputs:float array) (outputs:float array) =
   let weightsp = pin weights
   let proc_inputsp = pin proc_inputs
   let outputsp = pin outputs
   vectmat (NativePtr.ofNativeInt (~~proc_inputsp),NativePtr.ofNativeInt (~~weightsp),NativePtr.ofNativeInt (~~outputsp),proc_inputs.Length,weights.Length,outputs.Length)
   !~proc_inputsp
   !~weightsp
   !~outputsp   
   
 let vect_mat_dgemv  (weights:float array) (proc_inputs:float array) (outputs:float array) =
   let weightsp = pin weights
   let proc_inputsp = pin proc_inputs
   let outputsp = pin outputs
   
   dgemv('T',proc_inputs.Length,outputs.Length,1.0,~~~weightsp,proc_inputs.Length,~~~proc_inputsp,1,0.0,~~~outputsp,1)

   !~proc_inputsp
   !~weightsp
   !~outputsp   
   
