###############################################################################
# (c) Copyright 2019, Carlos Eduardo de Andrade.
# (c) 2026, Luis H. P. Mendes — adapted for NS-BRKGA
# All Rights Reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
###############################################################################

###############################################################################
# Required tools
###############################################################################

REQUIRED_TOOLS := doxygen doxyrest sphinx-build

$(foreach tool,$(REQUIRED_TOOLS),\
    $(if $(shell command -v $(tool) 2>/dev/null),,\
        $(error `$(tool)` not found in PATH. Please install `$(tool)`.)\
    )\
)

ifndef DOXYREST_DIR
$(error DOXYREST_DIR is not set. Export it before calling make, e.g.: export DOXYREST_DIR=/usr/local)
endif

################################################################################
## Build Rules
################################################################################

.PHONY: all clean doxygen doxyrest sphinx publish

all: doxygen doxyrest publish
	@echo "Documentation built successfully. Output: ../docs"

doxygen:
	doxygen Doxyfile

doxyrest:
	doxyrest -c doxyrest-config.lua

publish:
	rm -rf ../docs
	cp -a -f build/html_doxygen ../docs

clean:
	rm -rf build
