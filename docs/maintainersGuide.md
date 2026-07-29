## Creating a Release

0. select the new version number following https://semver.org/:

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
>    - MAJOR version when you make incompatible API changes
>    - MINOR version when you add functionality in a backward compatible manner
>    - PATCH version when you make backward compatible bug fixes
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

1. create a github issue documenting significant release changes; review the commit log and closed issues to find them

```
This issue is to document functionality and features added to MeshFields since the #.#.# release (SHA1):

New functionality or feature support:

- <feature> (SHA1,issueNumber)
- ...

Bug Fixes:

- <feature> (SHA1,issueNumber)
- ...

Other Updates and Improvements:

- <feature> (SHA1,issueNumber)
- ...
```

2. apply the issue/PR label 'v#.#.#' to significant issues and PR that are part of the release

3. increase the meshfields version # in CMakeLists.txt in the `main` branch
4. commit; include the issue # in the commit message

```
meshfields version v#.#.#                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                             
see issue #<###>
```

5. push
6. create the tag `git tag -a v#.#.# -m "meshfields version v#.#.#"`
7. push the tag `git push origin v#.#.#`

