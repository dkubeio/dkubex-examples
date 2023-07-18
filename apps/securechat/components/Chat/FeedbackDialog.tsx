import React, { ChangeEventHandler, FC, MouseEventHandler, useContext, useEffect, useReducer, useRef, useState } from 'react';

import { useTranslation } from 'next-i18next';

import { useCreateReducer } from '@/hooks/useCreateReducer';

import { getSettings, saveSettings } from '@/utils/app/settings';

import { Settings } from '@/types/settings';

import HomeContext from '@/pages/api/home/home.context';
import { IconThumbDown, IconThumbUp } from '@tabler/icons-react';
import { getEndpoint } from '@/utils/app/api';
import { Message } from '@/types/chat';
import { updateConversation } from '@/utils/app/conversation';

interface Props {
    open: boolean;
    onClose: () => void;
    isPositive: boolean;
    message: Message;
    // apiKey?: string;
    setIsPositiveFeedback: (v: boolean) => void;
    setSubmittingFeedback: (v: boolean) => void;
}

const POSITIVE_PLACEHOLDER = 'What do you like about the response?';
const NEGATIVE_PLACEHOLDER = 'What was the issue with the response? How could it be improved?';

const CheckboxOptions = [
    {
        id: '1',
        label: 'This is harmful / unsafe'
    },
    {
        id: '2',
        label: "This isn't true"
    },
    {
        id: '3',
        label: "This isn't helpful"
    }
];

export const FeedbackDialog: FC<Props> = ({ open, onClose, isPositive = true, message, setSubmittingFeedback, setIsPositiveFeedback }) => {
    const { t } = useTranslation('settings');
    const settings: Settings = getSettings();
    const [labelIds, setLabelIds] = useState<string[]>([]);
    const { state, dispatch } = useCreateReducer<Settings>({
        initialState: settings,
    });
    // const { dispatch: homeDispatch } = useContext(HomeContext);
    const {
        state: { selectedConversation, conversations, apiKey, messageIsStreaming },
        dispatch: homeDispatch,
    } = useContext(HomeContext);
    const modalRef = useRef<HTMLDivElement>(null);
    const [textValue, setTextValue] = useState('')

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setTextValue(e?.target?.value);
    }

    useEffect(() => {
        const handleMouseDown = (e: MouseEvent) => {
            if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
                window.addEventListener('mouseup', handleMouseUp);
            }
        };

        const handleMouseUp = (e: MouseEvent) => {
            window.removeEventListener('mouseup', handleMouseUp);
            onClose();
        };

        window.addEventListener('mousedown', handleMouseDown);

        return () => {
            window.removeEventListener('mousedown', handleMouseDown);
        };
    }, [onClose]);

    const handleSave = () => {
        homeDispatch({ field: 'lightMode', value: state.theme });
        saveSettings(state);
    };
    const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const id = e.target.name;
        if (labelIds.includes(id)) {
            setLabelIds(v => v.filter(i => i !== id));
        } else {
            setLabelIds(v => [...v, id]);
        }
    }
    const updateFeedback = () => {
        const newConversation: any = {
            ...selectedConversation, messages: selectedConversation?.messages.map(i => {
                if (message?.id === i?.id) {
                    return ({
                        ...i,
                        feedback: isPositive
                    })
                }
                return i
            })
        }
        const { single, all } = updateConversation(
            newConversation,
            conversations,
        );
        homeDispatch({ field: 'selectedConversation', value: single });
        homeDispatch({ field: 'conversations', value: all });
    }
    const handleFeedbackChange = async () => {
        const endpoint = getEndpoint(null, 'feedback')
        try {
            setSubmittingFeedback(true);
            const body = {
                key: apiKey,
                request_id: message?.id ?? '',
                feedback: isPositive === true ? 'positive' : 'negative',
                message: `${textValue ? textValue + '\n' : ''}${labelIds.map(i => CheckboxOptions.find(ele => ele.id === i)?.label).join('\n')}`
            }
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body),
            });
            //   console.log('hell data', body);
            //   setIsPositiveFeedback(isPositive);
            if (response.status === 200) {
                setIsPositiveFeedback(isPositive);
                updateFeedback();
            }

        } catch (e) {
            console.log("Error while setting feedback ", e);
        }
        setSubmittingFeedback(false);
    }
    // Render nothing if the dialog is not open.
    if (!open) {
        return <></>;
    }

    // Render the dialog.
    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="fixed inset-0 z-10 overflow-hidden">
                <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
                    <div
                        className="hidden sm:inline-block sm:h-screen sm:align-middle"
                        aria-hidden="true"
                    />

                    <div
                        ref={modalRef}
                        className="dark:border-netural-400 inline-block max-h-[400px] transform overflow-y-auto rounded-lg border border-gray-300 bg-white px-4 pt-5 pb-4 text-left align-bottom shadow-xl transition-all dark:bg-[#202123] sm:my-8 sm:max-h-[600px] sm:w-full sm:max-w-lg sm:p-6 sm:align-middle"
                        role="dialog"
                    >
                        <div className='flex flex-row justify-start gap-3  '>


                            <div className="text-lg pb-4 font-bold text-black dark:text-neutral-200">
                                {t('Provide additional Feedback')}
                            </div>
                            {isPositive ?
                                <div
                                    className={"text-green-500 dark:text-green-600"}
                                >
                                    <IconThumbUp size={20} />
                                </div> :
                                <div
                                    className={"text-red-500 dark:text-red-600"}
                                >
                                    <IconThumbDown size={20} />
                                </div>}
                        </div>
                        <textarea id="message" rows={6} className="mb-4 block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder={isPositive ? POSITIVE_PLACEHOLDER : NEGATIVE_PLACEHOLDER} value={textValue} onChange={handleChange} />

                        {
                            !isPositive && CheckboxOptions.map((i) => (
                                <div className="flex items-center mb-2" key={i.id}>
                                    <input checked={labelIds.includes(i.id)} name={i.id} id={`default-checkbox-${i.id}`} type="checkbox" value="" className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600" onChange={handleCheckboxChange} />
                                    <label htmlFor={`default-checkbox-${i.id}`} className="ml-2 text-sm font-medium text-gray-900 dark:text-gray-300">{i.label}</label>
                                </div>
                            ))
                        }

                        <div className='flex flex-row justify-end gap-2'>
                            <button
                                type="button"
                                className="base-1/4 px-2 py-1 mt-4 border rounded-lg shadow border-neutral-500 text-neutral-900 hover:bg-neutral-100 focus:outline-none dark:border-neutral-800 dark:border-opacity-50 dark:bg-white dark:text-black dark:hover:bg-neutral-300 text-xs"
                                onClick={() => {
                                    onClose();
                                }}
                            >
                                {t('Cancel')}
                            </button>
                            <button
                                type="button"
                                className="base-1/4 px-2 py-1 mt-4 border rounded-lg shadow border-neutral-500 text-neutral-900 hover:bg-neutral-100 focus:outline-none dark:border-neutral-800 dark:border-opacity-50 dark:bg-white dark:text-black dark:hover:bg-neutral-300 text-xs"
                                onClick={() => {
                                    // handleSave();
                                    handleFeedbackChange();
                                    onClose();
                                }}
                            >
                                {t('Submit feedback')}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
