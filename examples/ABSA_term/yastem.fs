module YaStem
open System
open System.IO
open System.Diagnostics

let ya_stem (word:string) =
 File.WriteAllText("sfile.txt",word)
 let startInfo = new ProcessStartInfo();
 startInfo.FileName <- "mystem.exe"
 startInfo.Arguments <- "sfile.txt soutfile.txt -c -l"
 startInfo.CreateNoWindow <- true
 startInfo.WindowStyle <- ProcessWindowStyle.Hidden
 let p = Process.Start(startInfo)
 p.WaitForExit()
 let out = File.ReadAllText("soutfile.txt")
 let o = ((((out.Replace("{","")).Replace("}","")).Replace("?","")).Split([|"|"|],StringSplitOptions.RemoveEmptyEntries)).[0]
 o.Trim()



(*let ()  =
 let s= (ya_stem "стиральные").Trim()
 Console.WriteLine(s)*)

