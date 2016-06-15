package = "dddt"
version = "scm-1"

source = {
   url = "https://github.com/zenna/dddt.git",
}

description = {
   summary = "Data Driven Data Types.",
   homepage = "",
   license = "MIT",
}

dependencies = {
   "torch >= 7.0",
   "autograd",
   "nn"
}

build = {
   type = "command",
   build_command = 'cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)',
   install_command = "cd build && $(MAKE) install"
}
