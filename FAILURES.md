# Failures

I think it's super useful to have a document describing what went wrong.


*   Examples were generated on normalized input:
    *   The fix for this was to denormalize and then
    *   Second round of fixes entirely removed the normalization
    *   Third round of fixes added back normalization but the descent step
        didn't normalize the inputs so we were getting a descent on the
        unnormalized input which made the attack less effective
*   WRN implementation doesn't match the paper: this is a minor nit because I
    was reading the paper and they mentioned that they were using the
    preactived shortcuts, but the official implementation does not do that.
*   Wasn't really sure how detach interacted with the code so I littered it
    everywhere
*   Removed a lot of in-place ops because autograd doesn't like them
    (memory allocation/buffer reuse is fast) and there's more bookeeping with
    the in-place for minimal savings
*   Average pool should have kernel 8 but instead used kernel 4
