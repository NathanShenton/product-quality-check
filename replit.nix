{ pkgs }:
let
  python = pkgs.python311;
in
{
  deps = [
    pkgs.glibcLocales
    python
    pkgs.python311Packages.streamlit
    pkgs.python311Packages.pandas
    pkgs.python311Packages.plotly
    pkgs.python311Packages.openai
  ];
}
