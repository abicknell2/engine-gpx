"runs a local flask application"
from datetime import datetime
import itertools
import json
import logging
import os
import queue
from threading import Thread
import time
from typing import Generator, cast

try:
    import numpy as np  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - numpy is available in prod but optional in tests
    np = None  # type: ignore[assignment]

from flask import (Flask, Response, has_request_context, jsonify, redirect, request)
from flask_httpauth import HTTPBasicAuth
import requests
import requests.auth
from sqlitedict import SqliteDict
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash

import api.constants as constants
from api.errors import ApiModelError
import api.oneshot as oneshot
from utils.error_helpers import (KNOWN_ERROR_CLASSES, get_root_exception, log_unexpected_error, should_mask_error)
import utils.logger as logger
from utils.types.shared import AcceptedTypes, ExcInfoType
from utils.unit_helpers import replace_units_to_process

SEND_API_VALUE_ERROR = True  # set to true to send the ValueError to the front end

app = Flask(__name__)

# initilize the basic authentication
auth = HTTPBasicAuth()

DB_FILE = "./user_db.sqlite"

ALLOWED_SOLVE_USERTYPES = ["full", "edu", "eval"]

SEND_RESULT_ERROR = "Issue encoding result. Please contact support."

# look for a pong time in the env
DEFAULT_PONG_TIME = 10  # seconds
PONG_TIME = float(os.getenv("COP_PONG_INT", DEFAULT_PONG_TIME))

if constants.THROW_HTTP_ERROR:
    if SEND_API_VALUE_ERROR:

        @app.errorhandler(ValueError)
        def handle_value_error(e: ValueError) -> tuple[Response, str]:
            logger.info(f"API|ValueError: {e}")
            response = jsonify({"error": str(e), "status": "Modeling Error"})
            response.status_code = 550

            # fix any pretty-printing issues esp. with units
            emsg = str(e)
            emsg = emsg.replace("**", "^")

            # Model Error

            logger.warn(f"API - MODEL ERROR | {emsg}")

            return response, f"{response.status_code} {emsg}"

    @app.errorhandler(ApiModelError)
    def handle_model_error(error: ApiModelError) -> tuple[Response, str]:
        logger.info(f"API|ModelError: {error}")
        "handles the api model errors and sends the specified message"
        resp = jsonify(error.to_dict()), f"{error.status_code} {error.message}"
        return resp


# if constants.THROW_HTTP_ERROR:
#     @app.errorhandler(EngineError)
#     def handle_engine_error(e):
#         # Engine Error
#         logging.info('API|EngineError: ' + str(e))
#         response = jsonify({'error' : str(e), 'status' : 'Modeling Error'})
#         response.status_code = 556

#         return response


# error handling https://flask.palletsprojects.com/en/1.1.x/errorhandling/
class SolverError(HTTPException):
    code = 555
    description = "Error generating the models"
    name = "Solver Error"


@app.errorhandler(SolverError)
def handle_solver_error(error: ApiModelError) -> Response:
    logger.info(f"API|Value Error: {error}")
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(HTTPException)
def handle_exception(e: HTTPException) -> Response:
    logger.info(f"API|HTTPException: {e}")
    """Return JSON instead of HTML for HTTP errors."""
    # replace the body with a JSON
    response = Response(
        response=json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }),
        status=e.code,
        headers=e.get_response().headers,
        content_type="application/json"
    )
    return response


# utility to find the user
def find_user(username: str) -> "None | dict[str, object]":
    "finds if the user is in one of the allowed classes"
    for ut in ALLOWED_SOLVE_USERTYPES:
        with SqliteDict(DB_FILE, tablename=ut) as udict:
            username = username.lower()
            if username in udict:
                logger.info(f"API | found User:{username} with Type:{ut}")
                return {
                    "username": username,
                    "usertype": ut,
                    "pwhash": udict[username],
                }
    # if there is no user found send
    return None


if getattr(constants, "USE_USERTYPE_LOGIN", False):
    logger.info("API | Use USERTYPE login")

    @auth.verify_password
    def verify_password(username: str, password: str) -> "str | bool":
        "verify the username and password on basic auth callback"
        # TODO: implement actual login flow
        # check the configuration for the correct database
        logger.info(f"API| login used for {username}")
        username = username.lower()
        udata = find_user(username)
        if udata is not None:
            if check_password_hash(udata["pwhash"], password):
                auth.usertype = udata["usertype"]
                logger.info(f"User {udata['username']} with type {auth.usertype}")
                return str(udata["username"])

        # if not logged in, send false
        logger.info(f"API | User:{username} rejected")
        return False

elif os.getenv("COP_NO_LOGIN"):

    @auth.verify_password
    def no_varification(username: str, password: str) -> str:
        username = username.lower()
        auth.username = username
        auth.usertype = "dev"
        return username

elif os.getenv("COP_EXTERN_LOGIN"):
    logger.info("using external login and verification")

    @auth.verify_password
    def extern_login(username: str, password: str) -> "str | bool":
        # get the authorization
        username = username.lower()
        basic = requests.auth.HTTPBasicAuth(username, password)
        # call the auth endpoint
        resp = requests.post("https://auth.linelab.app/api/authenticate", auth=basic)
        # look at the response
        if resp.status_code == requests.codes.ok:
            # put the data and return
            rd = resp.json()
            auth.usertype = rd.get("userType")
            return rd.get("username", username)
        else:
            # if not authorized, abort
            return False

else:
    # OLD define the authorizer
    logger.info("API | Use legacy login flow")

    @auth.verify_password
    def verify_password(username: str, password: str) -> "str | bool":  # noqa: F811
        "verify the username and password on basic auth callback"
        # TODO: implement actual login flow
        # check the configuration for the correct database
        logger.info(f"API| login used for {username}")
        with SqliteDict("./user_db.sqlite") as userdict:
            username = username.lower()
            if username in userdict:
                logger.info(f"API | username {username} found")
                if check_password_hash(userdict[username], password):
                    logger.info(f"API | password validated for {username}")
                    return username
            logger.info(f"API | user {username} rejected")

        return False


# route for basic verification
@app.route("/", methods=["POST"])
@auth.login_required
def home_auth() -> str:
    return auth.username()


# route for testing access
@app.route("/routetest", methods=["GET"])
def endpoint_access() -> str:
    # a simple test to make sure the endpoint is working from client computers
    return "Endpoint Access Successful"


def model_exception_hook(args: ExcInfoType) -> None:
    "handles the exception"
    exc_type, exc_value, exc_traceback = args
    raise exc_value


class SolverThread(Thread):

    def __init__(self, model: AcceptedTypes, pipeline: queue.Queue[AcceptedTypes]) -> None:
        logger.info("created a threaded solve")
        self.request = request
        self.auth = auth
        self.pipeline = pipeline
        self.model = model
        super().__init__()

    def run(self) -> None:
        logger.info("running a threaded solve")
        self.exc = None
        self.sol = None

        model: dict[str, AcceptedTypes] = self.model if isinstance(self.model, dict) else {}
        model_dict: dict[str, AcceptedTypes] = cast(dict[str, AcceptedTypes], model.get("model", {}))
        model_type: list[str] = cast(list[str], model_dict.get("type", []))

        try:
            if "rateHike" in model:
                logger.info("Run Ramp-Up Analysis")
                sol = oneshot.ramp_up_analysis(model)
            elif "bestworst" in model_type:
                logger.info("Solving best/worst")
                sol = oneshot.new_best_worst(model)
            elif 'system' in model_type or 'variants' in model_type:
                logging.info('solving a system')
                sol = oneshot.multi_product(self.model)
            else:
                logger.info("Solving simple model")
                sol = oneshot.run_one_shot(model)

            print('Result: ', sol)

            logger.info(f"solving complete at {time.time()}")
            self.ret = sol
            self.pipeline.put(sol)

        except Exception as e:
            root = get_root_exception(e)

            if self.auth and hasattr(self.auth, "username"):
                try:
                    username = self.auth.username()
                except Exception:
                    username = "anonymous"
            else:
                username = "anonymous"

            if isinstance(root, KNOWN_ERROR_CLASSES):
                self.error_uuid = log_unexpected_error(e, username=username)
            elif should_mask_error(e):
                self.error_uuid = log_unexpected_error(e, username=username)
            else:
                self.error_uuid = None

            logger.exception(f'{e} {self.error_uuid}')
            self.exc = e

    def join(self, timeout: "None | float" = None) -> object:  # type: ignore
        "modified join method to return error"
        super().join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


# def solve_thread(solpipe: queue.Queue, excpipe: queue.Queue, model: dict[str, object]) -> None:
#     logger.info("threaded solve as a function")
#     modeltype: str = model["model"]["type"]

#     try:
#         if "rateHike" in model:
#             logger.info("Run Ramp-Up Analysis")
#             sol = oneshot.ramp_up_analysis(model)
#         elif "bestworst" in modeltype:
#             logger.info("solving best/worst")
#             sol = oneshot.new_best_worst(model)
#         elif "system" in modeltype:
#             logger.info("solving a system")
#             sol = oneshot.multi_product(model)
#         else:
#             logger.info("solving simple model")
#             sol = oneshot.run_one_shot(model)

#         logger.info(f"solving complete at {time.time()}")
#         solpipe.put(sol)
#     except Exception as e:
#         logger.exception(e)
#         excpipe.put(e)


@app.route("/solve", methods=["POST"])
@auth.login_required
def threaded_solve() -> Response:
    # TODO:  check for stream header
    # HTTP_RESPONSETYPE = 'stream'
    sh: "None | str" = request.headers.environ.get("HTTP_RESPONSETYPE")  # noqa: F841
    # if not sh or sh != 'stream':
    #     print('request headers')
    #     print(str(request.headers))
    #     # redirected to a nonthreaded solve
    #     logging.info('stream header not found. redirect to non-threaded')
    #     return redirect('nonthreaded_solve', code=307)

    # copy the model from the results context
    model: AcceptedTypes = request.get_json(force=True)

    save_model_data_for_tests: bool = bool(
        request.args.get("save_model_data_for_tests") and os.getenv("SAVE_MODEL_DATA_FOR_TESTS")
    )
    if save_model_data_for_tests:
        if isinstance(model, dict) and isinstance(model.get("model"), dict):
            model_name: str = str(model["model"].get("modelName", "unnamed_model"))
        else:
            model_name = "unnamed_model"
        save_data(json.dumps(model), model_name, "input")

    model = replace_units_to_process(model)

    # generate a pipeline
    pipeline: queue.Queue[AcceptedTypes] = queue.Queue(maxsize=1)
    # solpipe: queue.Queue = queue.Queue(maxsize=1)
    # excpipe: queue.Queue = queue.Queue(maxsize=1)

    # create a threaded process
    th = SolverThread(model=model, pipeline=pipeline)
    # th = Thread(target=solve_thread, kwargs={'solpipe':solpipe, 'excpipe':excpipe, 'model':model})
    th.start()

    def gen(pipeline: queue.Queue[AcceptedTypes]) -> Generator[str, None, None]:

        while th.is_alive():
            time.sleep(PONG_TIME)  # try using a sleep (hopefully unblocks the thread)
            logger.info("pong")
            print("pong")
            yield "pong\n"

        # e = excpipe.get()
        # if e:
        #     logging.warn(f'Threaded Solve failed with: {e}')
        #     errtxt = str(e)
        #     if isinstance(e, ApiModelError):
        #         errtxt = e.message.strip()
        #         if not errtxt:
        #             errtxt = 'Model is infeasible. Check rate requirement and constraints.'
        #     resp = {'errors':[errtxt]}
        # else:
        #     resp = solpipe.get()

        # check to see if there is an error
        if th.exc:
            # raise th.exc
            logger.warn(f"Threaded Solve failed with: {th.exc}")
            errtxt: str = str(th.exc)

            no_error_text = False
            if isinstance(th.exc, ApiModelError):
                errtxt = th.exc.message.strip()
                if not errtxt:
                    no_error_text = True
                    error_msg = "Model is infeasible. Check rate requirement and constraints."

            if not no_error_text:
                if should_mask_error(th.exc):
                    error_msg = "An unexpected error occurred."
                else:
                    error_msg = str(errtxt)

            resp: AcceptedTypes = {"errors": [error_msg]}

        else:
            # get the end results
            logger.info(f"returning results at {time.time()}")
            # TODO:  get the solution attribute directly from the thread
            # resp = pipeline.get()
            resp = th.ret

        serializable_resp = ensure_json_serializable(resp)

        if save_model_data_for_tests:
            save_data(json.dumps(serializable_resp), model_name, "output")

        logging.debug("FRONTEND_ERROR: %s", serializable_resp.get("errors", 'No Errors'))
        yield json.dumps(serializable_resp)

    resp: Response = Response(gen(pipeline), mimetype="application/json")
    # resp = Response(gen(), mimetype='application/json')
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


def ensure_json_serializable(data: AcceptedTypes) -> AcceptedTypes:
    """Recursively convert values that `json.dumps` cannot handle."""
    if isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [ensure_json_serializable(item) for item in data]
    if isinstance(data, tuple):
        return [ensure_json_serializable(item) for item in data]
    if np is not None:
        if isinstance(data, np.generic):  # type: ignore[attr-defined]
            return data.item()
        if isinstance(data, np.ndarray):  # type: ignore[attr-defined]
            return data.tolist()
    return data


def save_data(data: str, model_name: str, data_type: str) -> None:
    """Save the provided data to a file in the extracted_data directory."""
    model_name = model_name.lower().replace(" ", "_")
    save_dir: str = os.path.join(os.path.dirname(__file__), "..", "test", "data", "json_data", model_name)
    os.makedirs(save_dir, exist_ok=True)

    file_name: str = f"{model_name}_results.json" if data_type == "output" else f"{model_name}.json"
    file_path: str = os.path.join(save_dir, file_name)

    # Check if the file already exists, meaning the model has already been processed
    if os.path.exists(file_path):
        logger.warn(f"File {file_path} already exists. Skipping save.")
        return

    try:
        # Ensure the data is properly formatted JSON
        if isinstance(data, str):
            data = json.loads(data)
        data = ensure_json_serializable(data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON data: {e}")
        raise

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


@app.route("/nonthreaded_solve", methods=["POST"])
@auth.login_required
def solve_one_shot() -> Response:
    model: dict[str, AcceptedTypes] = request.get("model", {})  # type: ignore
    model_type: list[str] = cast(list[str], model.get("type", []))
    # see this note on using 307 to forward the POST https://stackoverflow.com/questions/15473626/make-a-post-request-while-redirecting-in-flask
    # TODO:  see if we should record the input
    #       record the input by username and timestamp

    if not isinstance(model, dict):
        return Response(jsonify({"error": "Invalid model data"}), 400)

    try:
        if auth.usertype == "edu":
            # create the filepath to record the input as a json
            foldername: str = f"inputs-{auth.usertype}"
            filename: str = f"{auth.username()} {datetime.now()}"
            filename = filename.replace(".", "-") + ".json"
            filepath: str = os.path.join(".", foldername, filename)

            with open(filepath, "w") as f:
                json.dump(request.get_json(force=True), f, indent=4)

    except AttributeError:
        # this is using the old login. do not record
        pass

    if "rateHike" in model:
        logger.info("Run Ramp-Up Analysis")
        sol = oneshot.ramp_up_analysis(model)
        return jsonify(sol)
    if "bestworst" in model_type:
        logger.info("Flask redirect to bestworst")
        return cast(Response, redirect("bestworst", code=307))
    if 'system' in model_type or 'variants' in model_type:
        return cast(Response, redirect("system", code=307))

    sol = oneshot.run_one_shot(model)
    try:
        response = jsonify(sol)
    except TypeError as e:
        logger.error(f"API | Solved but fails to create result{e}")
        raise ValueError(SEND_RESULT_ERROR)

    return response


@app.route("/bestworst", methods=["POST"])
@auth.login_required
def one_shot_bw() -> Response:
    # print(request)

    # sol = oneshot.best_worst(request.get_json(force=True))
    sol = oneshot.new_best_worst(request.get_json(force=True))

    try:
        response = jsonify(sol)
    except TypeError as e:
        logger.error(f"API | Solved but fails to create result{e}")
        raise ValueError(SEND_RESULT_ERROR)
    # fix CORS
    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response


# TEMPORARY ENDPOINT FOR RAMP UP ANALYSIS
@app.route("/ramp", methods=["POST"])
def ramp_up() -> Response:
    "run the ramp up analysis"
    sol = oneshot.ramp_up_analysis(request.get_json(force=True))

    # return the response as a json
    return jsonify(sol)


@app.route("/system", methods=["POST"])
@auth.login_required
def one_shot_multiproduct() -> Response:
    "run the one shot for the mutli-product"
    sol = oneshot.multi_product(request.get_json(force=True))

    try:
        response = jsonify(sol)
    except TypeError as e:
        logger.error(f"API | Solved but fails to create result{e}")
        raise ValueError(SEND_RESULT_ERROR)

    return response


@app.route("/resources/v1/advanced_variables/manufacturing", methods=["POST"])
@auth.login_required
def mfg_adv_vars() -> Response:
    "gives the advanced manufacturing variables"
    resources = {
        "cell": ["queueing time", "flow time", "workstation count", "parallel capacity"],
        "line": ["flow time", "wip inventory"],
        "tool": ["count"],
        "feeder line": ["quantity"],
    }

    needs_type = {"cell": True, "line": False, "tool": True, "feeder line": True}
    zipped_resources = [list(zip([r] * len(pl[0]), pl)) for r, pl in resources.items()]
    keys = [{"category": r, "property": p, "needsType": needs_type[r]} for r, p in itertools.chain(*zipped_resources)]
    return jsonify(keys)


@app.route("/resources/v1/processes", methods=["POST"])
@auth.login_required
def process_library_catalog() -> Response:
    "gives the process libraries"
    return jsonify([
        "Milling",
        "Laser Cutting",
        "Press Forming",
        "Welding",
        "Spray Painting",
        "Automated Fiber Placement",
        "Autoclave Cure",
        "Resin Infusion",
        "Oven Cure",
    ])


@app.route("/pong")
def pingpong() -> Response:
    logger.info("pong health check")
    return Response("pong")
