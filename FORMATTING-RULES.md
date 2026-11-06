# Formatting Rules

The following rules apply to Python.  They follow the [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html), with the deviations
and refinements spelled out here.  Where this file and the Google guide
disagree, this file wins.  Defer to industry-standard best practices for code
other than Python.

The style workflow (`.github/workflows/style.yml`) enforces the mechanical
parts of these rules with `autopep8` and `flake8`, using a 120-character line
limit and flagging line breaks *before* binary operators (W503).

1. Lines longer than 120 characters should be broken across multiple lines.
   (This is UncertainSCI's convention; Google's guide uses 80.)  Lines that
   cannot reasonably be broken, such as long URLs, import statements, and
   tool-directive comments, are exempt.

   Prefer to break at the highest syntactic level, and use implicit line
   joining inside parentheses, brackets, and braces rather than backslashes.

   For function calls, break after the opening parenthesis, indent the
   arguments four spaces (a hanging indent), and put the closing parenthesis on
   its own line, aligned with the start of the statement.  If all of the
   arguments fit on one continuation line, they may share it; otherwise put each
   argument on its own line and end the list with a trailing comma.  The
   function call

   ```python
   this_is_an_example_of_a_long_line(some_very_long_argument_1, some_very_long_argument_2, some_very_long_argument_3, some_very_long_argument_4, some_very_long_argument_5)
   ```

   should be broken like:

   ```python
   this_is_an_example_of_a_long_line(
       some_very_long_argument_1,
       some_very_long_argument_2,
       some_very_long_argument_3,
       some_very_long_argument_4,
       some_very_long_argument_5,
   )
   ```

   The same hanging-indent-plus-trailing-comma layout applies to long list,
   tuple, dict, and set literals and to function definitions.

   For long non-function-call expressions, use parentheses and keep each
   continued expression element on its own line.  Binary operators stay at the
   *end* of continued lines (break after the operator, never before it).  The
   expression

   ```python
   some_very_long_result_name = some_very_long_argument_1 + some_very_long_argument_2 + some_very_long_argument_3 + some_very_long_argument_4 + some_very_long_argument_5
   ```

   should be broken like:

   ```python
   some_very_long_result_name = (
       some_very_long_argument_1 +
       some_very_long_argument_2 +
       some_very_long_argument_3 +
       some_very_long_argument_4 +
       some_very_long_argument_5
   )
   ```

2. A line ending with a colon (`:`) must not be followed by a blank line.  The
   next line should be executable code, or a docstring where docstrings are
   valid.  A standalone comment may follow the colon (see rule 5), but a blank
   line may not come between the colon and that comment.

   Top-level function and class definitions are separated by two blank lines;
   methods inside a class are separated by one.

   For classes, one blank line separates the class docstring from the first
   method or class attribute.  A class without a docstring has no blank line
   after the `class` line.  Blank lines inside a multi-line docstring are
   allowed.

   The following are *bad* examples:

   ```python
   def my_function(*args):

       """Does something!"""
       _my_function_internals(*args)
   ```

   ```python
   if my_condition:

       do_something()
   ```

   ```python
   class MyClass:

       """Containerizes something useful."""

       def __init__(self):
           _do_init(self)
   ```

   ```python
   class MyClass:

       def __init__(self):
           _do_init(self)
   ```

   ```python
   class MyClass:
       """Containerizes something useful."""
       def __init__(self):
           _do_init(self)
   ```

   The following are *good* examples:

   ```python
   def my_function(*args):
       _my_function_internals(*args)
   ```

   ```python
   def my_function(*args):
       """Does something."""
       _my_function_internals(*args)
   ```

   ```python
   def my_function(*args):
       """Does something.

       Args:
           *args: Some arguments.
       """
       _my_function_internals(*args)
   ```

   ```python
   if my_condition:
       do_something()  # This is a non-trivial comment about do_something.
   ```

   ```python
   class MyClass:
       """Containerizes something useful."""

       def __init__(self):
           _do_init(self)
   ```

   ```python
   class MyClass:
       def __init__(self):
           _do_init(self)
   ```

3. Within a function body, never use more than one contiguous blank line.  A
   single blank line may be used to separate substantial logical sections.
   Short, straight-line functions should generally not contain blank lines, and
   a blank line between a docstring and the first statement is permitted but
   not required.

   For example

   ```python
   def my_function(*args):
       intermediate = _my_function_step_1(*args)


       result = _my_function_step_2(intermediate)


       return result
   ```

   should be replaced by:

   ```python
   def my_function(*args):
       intermediate = _my_function_step_1(*args)
       result = _my_function_step_2(intermediate)
       return result
   ```

   For very simple data transformations, prefer returning the result directly:

   ```python
   def my_function(*args):
       intermediate = _my_function_step_1(*args)
       return _my_function_step_2(intermediate)
   ```

4. Docstrings use triple double quotes (`"""`).  A docstring begins with a
   one-line summary that fits on a single line and ends with a period.  Write
   the summary in the descriptive (third-person) style, e.g. "Fetches rows from
   the table.", not the imperative style ("Fetch rows ...").  Do not use
   exclamation marks.  For a multi-line docstring, follow the summary with a
   blank line, then the details, and put the closing quotes on their own line.

   Public functions and methods document their parameters, return values, and
   raised exceptions using the Google section names `Args:`, `Returns:` (or
   `Yields:`), and `Raises:`, in that order, with each entry indented under its
   header.  Do not use `Arguments:` or other variants.

   ```python
   def scale(values, factor):
       """Multiplies each value by a constant factor.

       Args:
           values: Sequence of numbers to scale.
           factor: Multiplier applied to every entry of `values`.

       Returns:
           A list containing the scaled values.

       Raises:
           ValueError: If `factor` is not finite.
       """
       ...
   ```

5. Comments use exactly one space after the `#`.  Block comments (standalone
   comments on their own lines) should be full, capitalized sentences ending
   with a period.  Inline comments (on the same line as code) are separated from
   the code by at least two spaces, should be brief, and may be sentence
   fragments, though full sentences are preferred.

   Bad comments:

   ```python
   #bad comment
   ```

   ```python
   #  another bad comment
   ```

   ```python
   x = 1 # Too close to the code.
   ```

   A good block comment:

   ```python
   # This is a good comment.
   ```

   `TODO` comments use the form `TODO: <issue link or ID> - <sentence>`, so that
   each one can be tracked:

   ```python
   # TODO: github.com/SCIInstitute/UncertainSCI/issues/123 - Support complex-valued inputs.
   ```

   New `TODO` comments must include the issue link or ID.  Older `TODO`
   comments that predate this rule may omit it (`# TODO: <sentence>`) until an
   issue is filed for them.  Use `TODO`, not `FIXME` or `XXX`.

   Directives read by tools or by the interpreter are exempt from the sentence
   and spacing rules above, and must keep whatever form the tool requires.
   Examples are shebang lines, encoding declarations, `# noqa: ...`,
   `# type: ...`, and `# pylint: ...`.

   Standalone comments immediately after a line ending with a colon, or
   immediately after a docstring, are discouraged but not forbidden.  Prefer
   moving that information into the docstring or closer to the code it
   describes.  When such a comment is used, it must directly follow the colon
   line (no blank line between them), as required by rule 2.

6. Code that must be disabled should *not* be commented out with triple quotes
   or `#`.  Remove it, or place it behind an explicit condition when it must
   remain available.

   Bad:

   ```python
   def my_function():
       # return 'Hello world!'
       return None
   ```

   ```python
   def my_function():
       """
       return 'Hello world!'
       """
       return None
   ```

   Acceptable when the disabled path is intentionally retained:

   ```python
   def my_function(use_greeting):
       if use_greeting:
           return 'Hello world!'
       return None
   ```

   Better when the disabled code is no longer needed:

   ```python
   def my_function():
       return None
   ```

7. Naming follows the Google guide, with the following conventions:

   - Functions, methods, variables, arguments, and modules use `snake_case`
     (`lower_with_under`).
   - Classes use `CapWords` (e.g. `MyClass`, `PolynomialChaosExpansion`).
   - Constants use `UPPER_WITH_UNDER`.
   - Internal (non-public) names begin with a single leading underscore.
   - Names taken from established mathematical notation (e.g. `N`, `M`, `A`,
     `ab`) may keep that notation, including capitals, when they match the
     formulas or references the code implements.

   Existing public names that do not follow these conventions should be
   renamed only with a deprecation path, since doing so changes the public API.

8. Defer to the Google Python Style Guide, and then to general best practices
   (PEP 8), for rules not otherwise specified here.
