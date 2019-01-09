with import <nixpkgs> { };

(python36.buildEnv.override {
  extraLibs = with pkgs.python36Packages; [
    numpy
    scipy

    #Utilities
    yapf
    python-language-server
  ];

  ignoreCollisions = true;
}).env
