# Formatting Rules

The following rules apply to Python.  Defer to industry-standard best practices
for code other than Python.

1. Lines longer than 120 characters should be broken across multiple lines.

   For function calls, put each argument on its own line.  The function call

   ```python
   thisIsAnExampleOfALongLine(someVeryLongArgument1, someVeryLongArgument2, someVeryLongArgument3, someVeryLongArgument4, someVeryLongArgument5)
   ```

   should be broken like:

   ```python
   thisIsAnExampleOfALongLine(
       someVeryLongArgument1,
       someVeryLongArgument2,
       someVeryLongArgument3,
       someVeryLongArgument4,
       someVeryLongArgument5
   )
   ```

   For long non-function call expressions, use parentheses and keep each
   continued expression element on its own line.  Binary operators should stay
   at the end of continued lines.  The expression

   ```python
   someVeryLongResultName = someVeryLongArgument1 + someVeryLongArgument2 + someVeryLongArgument3 + someVeryLongArgument4 + someVeryLongArgument5
   ```

   should be broken like:

   ```python
   someVeryLongResultName = (
       someVeryLongArgument1 +
       someVeryLongArgument2 +
       someVeryLongArgument3 +
       someVeryLongArgument4 +
       someVeryLongArgument5
   )
   ```

2. Lines of code ending with a colon (`:`) should be followed immediately by
   executable code, or by a docstring where docstrings are valid.  The
   class-definition exception is that one blank line may separate a class
   docstring from the first method or class attribute.  Blank lines inside a
   multi-line docstring are allowed.

   The following are *bad* examples:

   ```python
   def myFunction(*args):

       """myFunction does something!"""

       _myFunctionInternals(*args)
   ```

   ```python
   def myFunction(*args):
       """myFunction does something!"""

       _myFunctionInternals(*args)
   ```

   ```python
   def myFunction(*args):
       """myFunction does something!

       Arguments:
           args: Some arguments.
       """

       _myFunctionInternals(*args)
   ```

   ```python
   if myCondition:


       # This is a trivial comment about doSomething.

       doSomething()
   ```

   ```python
   class MyClass:

       """MyClass containerizes something useful."""

       def __init__(self):
           _doInit(self)
   ```

   ```python
   class MyClass:

       def __init__(self):
           _doInit(self)
   ```

   The following are *good* examples:

   ```python
   def myFunction(*args):
       _myFunctionInternals(*args)
   ```

   ```python
   def myFunction(*args):
       """myFunction does something!"""
       _myFunctionInternals(*args)
   ```

   ```python
   def myFunction(*args):
       """myFunction does something!

       Arguments:
           args: Some arguments.
       """
       _myFunctionInternals(*args)
   ```

   ```python
   if myCondition:
       doSomething()  # This is a non-trivial comment about doSomething.
   ```

   ```python
   class MyClass:
       """MyClass containerizes something useful."""

       def __init__(self):
           _doInit(self)
   ```

   ```python
   class MyClass:
       def __init__(self):
           _doInit(self)
   ```

3. Within a function body, never use more than one contiguous blank line.  A
   single blank line may be used to separate substantial logical sections, but
   short straight-line functions should not contain blank lines.

   For example

   ```python
   def myFunction(*args):
       intermediate = _myFunctionStep1(*args)


       result = _myFunctionStep2(intermediate)


       return result
   ```

   should be replaced by:

   ```python
   def myFunction(*args):
       intermediate = _myFunctionStep1(*args)
       result = _myFunctionStep2(intermediate)
       return result
   ```

   For very simple data transformations, prefer returning the result directly.  For
   example:

   ```python
   def myFunction(*args):
       intermediate = _myFunctionStep1(*args)
       return _myFunctionStep2(intermediate)
   ```

4. Comments should be full sentences, with exactly one space following the `#`.

   Bad comments:

   ```python
   #bad comment
   ```

   ```python
   #  another bad comment
   ```

   A good comment:

   ```python
   # This is a good comment.
   ```

   `TODO` comments and other inline memos should use the same spacing and
   sentence style:

   ```python
   # TODO: This is a complete sentence directing some future reader.
   ```

5. Standalone `#` comments immediately after a line ending with a colon (`:`) or
   immediately after a docstring are discouraged but not forbidden.  Prefer
   moving that information into the docstring or closer to the code it
   describes.

6. Code that must be disabled should *not* be commented out with triple quotes
   or `#`.  Remove it, or place it behind an explicit condition when it must
   remain available.

   Bad:

   ```python
   def myFunction():
       # return 'Hello world!'
       return None
   ```

   ```python
   def myFunction():
       """
       return 'Hello world!'
       """
       return None
   ```

   Acceptable when the disabled path is intentionally retained:

   ```python
   def myFunction(useGreeting):
       if useGreeting:
           return 'Hello world!'
       return None
   ```

   Better when the disabled code is no longer needed:

   ```python
   def myFunction():
       return None
   ```

7. Defer to general best practices for rules not otherwise specified here.
   These rules should be compatible with, or reinforce, common Python style
   conventions.
