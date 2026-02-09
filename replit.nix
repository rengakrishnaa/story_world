{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.ffmpeg
    pkgs.git
    pkgs.postgresql
    pkgs.zlib
  ];
}
