namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("Meanotek-NeuText")>]
[<assembly: AssemblyProductAttribute("Meanotek-NeuText")>]
[<assembly: AssemblyDescriptionAttribute("Deep learning library for text processing")>]
[<assembly: AssemblyVersionAttribute("1.0")>]
[<assembly: AssemblyFileVersionAttribute("1.0")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "1.0"
