using System;

namespace TVMRuntime
{
    public static class Binding
    {
        public static Runtime runtime { get; } = new Runtime();

        public static Module module { get; } = new Module();
    }
}
