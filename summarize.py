from pathlib import Path
from config import get_config, latest_weights_file_path
from train import get_model
from tokenizers import Tokenizer
import torch
from pyvi import ViTokenizer


def summarize(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()

    sentence = ViTokenizer.tokenize(sentence)
    tokenizer = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["language"])))
    )
    print(str(Path(config["tokenizer_file"].format(config["language"]))))
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    print("ok1")
    model.load_state_dict(state["model_state_dict"])
    print("ok")
    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer.encode(sentence)
        source = torch.cat(
            [
                torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer.token_to_id("[PAD]")]
                    * (config["src_len"] - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        source = source.unsqueeze(0)
        source_mask = (
            (source != tokenizer.token_to_id("[PAD]"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )
        print(source.shape)
        print(source_mask.shape)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = (
            torch.empty(1, 1)
            .fill_(tokenizer.token_to_id("[SOS]"))
            .type_as(source)
            .to(device)
        )

        # Print the source sentence and target start prompt

        # Generate the translation word by word
        while decoder_input.size(1) < config["tgt_len"]:
            # build mask for target and calculate output
            decoder_mask = (
                (
                    torch.triu(
                        torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                        diagonal=1,
                    )
                    == 0
                )
                .type(torch.int)
                .type_as(source_mask)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device),
                ],
                dim=1,
            )

            # print the translated word
            print(f"{tokenizer.decode([next_word.item()])}", end=" ")

            # break if we predict the end of sentence token
            if next_word == tokenizer.token_to_id("[EOS]"):
                break

    # convert ids to tokens
    return tokenizer.decode(decoder_input[0].tolist())


# read sentence from argument
summarize(
    f'AFP cho biết 7 vụ nổ xảy ra gần như đồng_thời nhắm vào các mục_tiêu là trạm xe_buýt , bệnh_viện và các khu_vực có nhiều thường_dân tại các thành_phố ven biển là Jableh và Tartus . Đây là các thành_phố tương_đối cách_ly khỏi cuộc nội_chiến kéo_dài suốt 5 năm qua tại Syria . Các cuộc tấn_công bắt_đầu bằng 3 vụ nổ tại một trạm xe_buýt đông_đúc ở Tartus nơi có một cơ_sở hải_quân của Nga . Đài_Quan_sát Nhân_quyền Syria cho biết chiếc bom xe phát_nổ đầu_tiên . Khi mọi người bắt_đầu đến hiện_trường vụ đánh bom này thì hai kẻ đánh bom tự_sát đã kích_hoạt đai bom . Đài cũng cho biết có 4 vụ nổ khác , một quả bom xe và 3 kẻ đánh bom liều chết , nhằm vào một trạm xe_buýt , một bệnh_viện và một trạm phát_điện . " Tôi bị sốc . Đây là lần đầu_tiên tôi nghe thấy âm_thanh như thế . Tôi nghĩ rằng cuộc_chiến đã qua và tôi có_thể đi bộ an_toàn . Tôi sốc khi thấy rằng chúng_tôi vẫn đang ở trung_tâm của cuộc_chiến " - sinh_viên Mohsen_Zayyoud tại Jableh chia_sẻ . " Đây là lần đầu_tiên chúng_tôi nghe thấy tiếng nổ tại Tartus và là lần đầu_tiên chúng_tôi thấy người chết hoặc các thi_thể ở đây " - nhân_viên ngân_hàng Shady Osman sống tại Tartus kể lại . Các cuộc tấn_công đẫm máu nhắm vào thành_lũy của chế_độ tổng_thống Bashar_al - Assad xảy ra trong bối_cảnh IS đang đối_mặt với sức_ép ở cả Iraq và Syria . Đài_Quan_sát Nhân_quyền Syria cho biết có 100 người thiệt_mạng trong các vụ tấn_công tại thành_phố Jableh trong khi các vụ tấn_công tại Tartus khiến 48 người khác chết , bao_gồm ít_nhất 8 trẻ_em . Người đứng đầu Đài_Quan_sát Nhân_quyền Syria Rami Abdel Rahman nói rằng họ " không nghi_ngờ gì rằng các cuộc tấn_công này là những vụ đẫm máu nhất " tại hai thành_phố trên kể từ khi chiến_tranh nổ ra ở Syria . IS đã lên_tiếng nhận trách_nhiệm thông_qua hãng thông_tấn Amaq của tổ_chức khủng_bố này , tuyên_bố rằng các tay súng chiến_đấu của tổ_chức đã tấn_công " những nơi tập_trung Alawite " tại Jableh và Tartus . Alawite là một giáo_phái_thiểu_số gồm những con_người ca_ngợi tổng_thống Assad . Trước_nay IS không hiện_diện nhiều tại các tỉnh ven biển Syria do khu_vực này là nơi các chiến_binh thuộc chi_nhánh địa_phương Al - Nusra Front của Al - Qaeda hoạt_động nổi_bật hơn . Tuy_nhiên IS đang đề_nghị thành_lập một " wilayat al - Sahel " hay còn gọi là một tỉnh Hồi_giáo cho khu_vực ven biển Syria mà trước giờ tổ_chức này ít đụng đến . Tổng_Thư_ký LHQ Ban Ki Moon lên_án " các cuộc tấn_công khủng_bố " . Tổ_chức Giám_sát Nhân_quyền cũng lên_án các vụ đánh bom , cáo_buộc IS nhắm vào thường_dân . Nga lên_án cuộc tấn_công , nói rằng các vụ khủng_bố này " một lần nữa đã chứng_minh tình_trạng mong_manh của Syria và sự cần_thiết phải có một biện_pháp mạnh để tái_khởi_động lại cuộc đàm_phán hòa bình " . Pháp gọi các vụ đánh bom tại hai thành_phố trên là " hành_động ghê_tởm " .'
)
